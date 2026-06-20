import os
import torch
import copy
import math
import jax
import functools as fn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.distributed as dist
from collections import defaultdict
from multiflow.data import so3_utils, all_atom
from multiflow.data import utils as du
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from torch import autograd
from torch.distributions.categorical import Categorical
from torch.distributions.binomial import Binomial
from colabdesign import mk_afdesign_model, clear_mem
from multiflow.analysis import utils as au
from biotite.sequence.io import fasta
from multiflow.data.residue_constants import restypes, restypes_with_x
from Bio import PDB
from multiflow.experiments.target_binder_metrics import calculate_all_metrics


def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)


def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)


def _masked_categorical(num_batch, num_res, device):
    return torch.ones(
        num_batch, num_res, device=device) * du.MASK_TOKEN_INDEX


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


def _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask):
    return aatypes_t * diffuse_mask + aatypes_1 * (1 - diffuse_mask)


class TargetBinderInterpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._aatypes_cfg = cfg.aatypes
        self._sample_cfg = cfg.sampling
        self._igso3 = None

        self.num_tokens = 21 if self._aatypes_cfg.interpolant_type == "masking" else 20

        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if cfg.beam_search:
            self.beam_search = True
            self.metric_model_kwargs = dict(
                protocol="binder",
                use_multimer=True,
                data_dir=cfg.alphafold2_params_dir,
                use_remat=False,
            )
            self.metric_model = None
            self.beam_search_freq = cfg.beam_search_freq
            self.beam_search_split_num = cfg.beam_search_split_num
            self.noise_scale = 0.1
            self.parser = PDB.PDBParser(QUIET=True)
        else:
            self.beam_search = False
            self.metric_model_kwargs = None
            self.metric_model = None
            self.beam_search_freq = cfg.sampling.num_timesteps
            self.beam_search_split_num = 1
            self.noise_scale = 0.0
            self.parser = None



    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        if self._trans_cfg.batch_ot:
            trans_0 = self._batch_ot(trans_0, trans_1, diffuse_mask)
        if self._trans_cfg.train_schedule == 'linear':
            trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        else:
            raise ValueError(
                f'Unknown trans schedule {self._trans_cfg.train_schedule}')
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]
    
    def _batch_ot(self, trans_0, trans_1, res_mask):
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        ) 
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)
        
        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        
        so3_schedule = self._rots_cfg.train_schedule
        if so3_schedule == 'exp':
            so3_t = 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif so3_schedule == 'linear':
            so3_t = t
        else:
            raise ValueError(f'Invalid schedule: {so3_schedule}')
        rotmats_t = so3_utils.geodesic_t(so3_t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def _corrupt_aatypes(self, aatypes_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)
        assert res_mask.shape == (num_batch, num_res)
        assert diffuse_mask.shape == (num_batch, num_res)

        if self._aatypes_cfg.interpolant_type == "masking":
            u = torch.rand(num_batch, num_res, device=self._device)
            aatypes_t = aatypes_1.clone()
            corruption_mask = u < (1 - t) # (B, N)

            aatypes_t[corruption_mask] = du.MASK_TOKEN_INDEX

            aatypes_t = aatypes_t * res_mask + du.MASK_TOKEN_INDEX * (1 - res_mask)

        elif self._aatypes_cfg.interpolant_type == "uniform":
            u = torch.rand(num_batch, num_res, device=self._device)
            aatypes_t = aatypes_1.clone()
            corruption_mask = u < (1-t) # (B, N)
            uniform_sample = torch.randint_like(aatypes_t, low=0, high=du.NUM_TOKENS)
            aatypes_t[corruption_mask] = uniform_sample[corruption_mask]

            aatypes_t = aatypes_t * res_mask + du.MASK_TOKEN_INDEX * (1 - res_mask)
        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")

        return _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask)

    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        aatypes_1 = batch['aatypes_1']

        # [B, N]
        res_mask = batch['res_mask']
        diffuse_mask = batch['diffuse_mask']
        num_batch, num_res = diffuse_mask.shape
        target_seq_len = batch['target_seq_len']
        binder_seq_len = batch['binder_seq_len']

        # [B, 1]
        if self._cfg.codesign_separate_t:
            def sample_separate_t():
                u = torch.rand((num_batch,), device=self._device)
                forward_fold_mask = (u < self._cfg.codesign_forward_fold_prop).float()
                inverse_fold_mask = (u < self._cfg.codesign_inverse_fold_prop + self._cfg.codesign_forward_fold_prop).float() * \
                    (u >= self._cfg.codesign_forward_fold_prop).float()

                normal_structure_t = self.sample_t(num_batch)
                inverse_fold_structure_t = torch.ones((num_batch,), device=self._device)
                normal_cat_t = self.sample_t(num_batch)
                forward_fold_cat_t = torch.ones((num_batch,), device=self._device)

                # If we are forward folding, then cat_t should be 1
                # If we are inverse folding or codesign then cat_t should be uniform
                cat_t = forward_fold_mask * forward_fold_cat_t + (1 - forward_fold_mask) * normal_cat_t

                # If we are inverse folding, then structure_t should be 1
                # If we are forward folding or codesign then structure_t should be uniform
                structure_t = inverse_fold_mask * inverse_fold_structure_t + (1 - inverse_fold_mask) * normal_structure_t

                so3_t = structure_t[:, None]
                r3_t = structure_t[:, None]
                cat_t = cat_t[:, None]

                return so3_t, r3_t, cat_t

            if self._cfg.noise_target:
                so3_t, r3_t, cat_t = sample_separate_t()
                target_so3_t, target_r3_t, target_cat_t = sample_separate_t()
            else:
                so3_t, r3_t, cat_t = sample_separate_t()
                target_so3_t = torch.ones((num_batch,), device=self._device)[:, None]
                target_r3_t = torch.ones((num_batch,), device=self._device)[:, None]
                target_cat_t = torch.ones((num_batch,), device=self._device)[:, None]

        else:
            if self._cfg.noise_target:
                t = self.sample_t(num_batch)[:, None]
                so3_t = t
                r3_t = t
                cat_t = t
                target_t = self.sample_t(num_batch)[:, None]
                target_so3_t = target_t
                target_r3_t = target_t
                target_cat_t = target_t
            else:
                t = self.sample_t(num_batch)[:, None]
                so3_t = t
                r3_t = t
                cat_t = t
                target_so3_t = torch.ones((num_batch, 1), device=self._device)[:, None]
                target_r3_t = torch.ones((num_batch, 1), device=self._device)[:, None]
                target_cat_t = torch.ones((num_batch, 1), device=self._device)[:, None]
        noisy_batch['binder_so3_t'] = so3_t
        noisy_batch['binder_r3_t'] = r3_t
        noisy_batch['binder_cat_t'] = cat_t
        noisy_batch['target_so3_t'] = target_so3_t
        noisy_batch['target_r3_t'] = target_r3_t
        noisy_batch['target_cat_t'] = target_cat_t

        all_one_diffuse_mask = torch.ones_like(diffuse_mask)
        target_binder_diffuse_mask = torch.ones_like(diffuse_mask)
        for i, (target_seq_len_i, binder_seq_len_i) in enumerate(zip(target_seq_len, binder_seq_len)):
            target_binder_diffuse_mask[i, :target_seq_len_i] = 0.0
            target_binder_diffuse_mask[i, target_seq_len_i:target_seq_len_i + binder_seq_len_i] = 1.0
        if self._trans_cfg.corrupt:
            binder_trans_t = self._corrupt_trans(
                trans_1, r3_t, res_mask, all_one_diffuse_mask)
            target_trans_t = self._corrupt_trans(
                trans_1, target_r3_t, res_mask, all_one_diffuse_mask)
            trans_t = binder_trans_t * target_binder_diffuse_mask[..., None] \
                        + target_trans_t * (1 - target_binder_diffuse_mask[..., None])
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError('NaN in trans_t during corruption')
        noisy_batch['trans_t'] = trans_t

        if self._rots_cfg.corrupt:
            binder_rotmats_t = self._corrupt_rotmats(rotmats_1, so3_t, res_mask, all_one_diffuse_mask)
            target_rotmats_t = self._corrupt_rotmats(rotmats_1, target_so3_t, res_mask, all_one_diffuse_mask)
            rotmats_t = binder_rotmats_t * target_binder_diffuse_mask[..., None, None] \
                        + target_rotmats_t * (1 - target_binder_diffuse_mask[..., None, None])
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError('NaN in rotmats_t during corruption')
        noisy_batch['rotmats_t'] = rotmats_t

        if self._aatypes_cfg.corrupt:
            binder_aatypes_t = self._corrupt_aatypes(aatypes_1, cat_t, res_mask, all_one_diffuse_mask)
            target_aatypes_t = self._corrupt_aatypes(aatypes_1, target_cat_t, res_mask, all_one_diffuse_mask)
            aatypes_t = binder_aatypes_t * target_binder_diffuse_mask \
                        + target_aatypes_t * (1 - target_binder_diffuse_mask)
        else:
            aatypes_t = aatypes_1
        noisy_batch['aatypes_t'] = aatypes_t
        if self._cfg.noise_target:
            noisy_batch['trans_sc'] = torch.zeros_like(trans_1)
            noisy_batch['aatypes_sc'] = torch.zeros_like(
                aatypes_1)[..., None].repeat(1, 1, self.num_tokens)
        else:
            noisy_batch['trans_sc'] = torch.cat([
                torch.cat([trans_1[i:i+1, :target_seq_len[i]], torch.zeros_like(trans_1[i:i+1, target_seq_len[i]:])], dim=1) for i in range(num_batch)
            ], dim=0)
            noisy_batch['aatypes_sc'] = torch.cat([
                torch.cat([aatypes_1[i:i+1, :target_seq_len[i]], torch.zeros_like(aatypes_1[i:i+1, target_seq_len[i]:])], dim=1) for i in range(num_batch)
            ], dim=0)[..., None].repeat(1, 1, self.num_tokens)

        noisy_batch['target_seq_len'] = target_seq_len
        noisy_batch['binder_seq_len'] = binder_seq_len
        noisy_batch['hotspot'] = batch['hotspot']

        return noisy_batch
    
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_vector_field(self, t, trans_1, trans_t):
        if self._trans_cfg.sample_schedule == 'linear':
            trans_vf = (trans_1 - trans_t) / (1 - t)
        elif self._trans_cfg.sample_schedule == 'vpsde':
            bmin = self._trans_cfg.vpsde_bmin
            bmax = self._trans_cfg.vpsde_bmax
            bt = bmin + (bmax - bmin) * (1-t) # scalar
            alpha_t = torch.exp(- bmin * (1-t) - 0.5 * (1-t)**2 * (bmax - bmin)) # scalar
            trans_vf = 0.5 * bt * trans_t + \
                0.5 * bt * (torch.sqrt(alpha_t) * trans_1 - trans_t) / (1 - alpha_t)
        else:
            raise ValueError(
                f'Invalid sample schedule: {self._trans_cfg.sample_schedule}'
            )
        return trans_vf

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t >= 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        # TODO: Add in SDE.
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)

    def _regularize_step_probs(self, step_probs, aatypes_t):
        batch_size, num_res, S = step_probs.shape
        device = step_probs.device
        assert aatypes_t.shape == (batch_size, num_res)

        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        # TODO replace with torch._scatter
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res),
            torch.arange(num_res, device=device).repeat(batch_size),
            aatypes_t.long().flatten()
        ] = 0.0
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res),
            torch.arange(num_res, device=device).repeat(batch_size),
            aatypes_t.long().flatten()
        ] = 1.0 - torch.sum(step_probs, dim=-1).flatten()
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        return step_probs

    def _aatypes_euler_step(self, d_t, t, logits_1, aatypes_t):
        # S = 21
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        if self._aatypes_cfg.interpolant_type == "masking":
            assert S == 21
            device = logits_1.device
            
            mask_one_hot = torch.zeros((S,), device=device)
            mask_one_hot[du.MASK_TOKEN_INDEX] = 1.0

            logits_1[:, :, du.MASK_TOKEN_INDEX] = -1e9

            pt_x1_probs = F.softmax(logits_1 / self._aatypes_cfg.temp, dim=-1) # (B, D, S)

            aatypes_t_is_mask = (aatypes_t == du.MASK_TOKEN_INDEX).view(batch_size, num_res, 1).float()
            step_probs = d_t * pt_x1_probs * ((1+ self._aatypes_cfg.noise*t) / ((1 - t))) # (B, D, S)
            step_probs += d_t * (1 - aatypes_t_is_mask) * mask_one_hot.view(1, 1, -1) * self._aatypes_cfg.noise

            step_probs = self._regularize_step_probs(step_probs, aatypes_t)

            return torch.multinomial(step_probs.view(-1, S), num_samples=1).view(batch_size, num_res)
        elif self._aatypes_cfg.interpolant_type == "uniform":
            assert S == 20
            assert aatypes_t.max() < 20, "No UNK tokens allowed in the uniform sampling step!"
            device = logits_1.device

            pt_x1_probs = F.softmax(logits_1 / self._aatypes_cfg.temp, dim=-1) # (B, D, S)

            pt_x1_eq_xt_prob = torch.gather(pt_x1_probs, dim=-1, index=aatypes_t.long().unsqueeze(-1)) # (B, D, 1)
            assert pt_x1_eq_xt_prob.shape == (batch_size, num_res, 1)

            N = self._aatypes_cfg.noise
            step_probs = d_t * (pt_x1_probs * ((1 + N + N * (S - 1) * t) / (1-t)) + N * pt_x1_eq_xt_prob )

            step_probs = self._regularize_step_probs(step_probs, aatypes_t)

            return torch.multinomial(step_probs.view(-1, S), num_samples=1).view(batch_size, num_res)
        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")

    def _aatypes_euler_step_purity(self, d_t, t, logits_1, aatypes_t):
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        assert S == 21
        assert self._aatypes_cfg.interpolant_type == "masking"
        device = logits_1.device

        logits_1_wo_mask = logits_1[:, :, 0:-1] # (B, D, S-1)
        pt_x1_probs = F.softmax(logits_1_wo_mask / self._aatypes_cfg.temp, dim=-1) # (B, D, S-1)
        # step_probs = (d_t * pt_x1_probs * (1/(1-t))).clamp(max=1) # (B, D, S-1)
        max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0] # (B, D)
        # bias so that only currently masked positions get chosen to be unmasked
        max_logprob = max_logprob - (aatypes_t != du.MASK_TOKEN_INDEX).float() * 1e9
        sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True) # (B, D)

        unmask_probs = (d_t * ( (1 + self._aatypes_cfg.noise * t) / (1-t)).to(device)).clamp(max=1) # scalar

        number_to_unmask = torch.binomial(count=torch.count_nonzero(aatypes_t == du.MASK_TOKEN_INDEX, dim=-1).float(),
                                          prob=unmask_probs)
        unmasked_samples = torch.multinomial(pt_x1_probs.view(-1, S-1), num_samples=1).view(batch_size, num_res)

        # Vectorized version of:
        # for b in range(B):
        #     for d in range(D):
        #         if d < number_to_unmask[b]:
        #             aatypes_t[b, sorted_max_logprobs_idcs[b, d]] = unmasked_samples[b, sorted_max_logprobs_idcs[b, d]]

        D_grid = torch.arange(num_res, device=device).view(1, -1).repeat(batch_size, 1)
        mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
        inital_val_max_logprob_idcs = sorted_max_logprobs_idcs[:, 0].view(-1, 1).repeat(1, num_res)
        masked_sorted_max_logprobs_idcs = (mask1 * sorted_max_logprobs_idcs + (1-mask1) * inital_val_max_logprob_idcs).long()
        mask2 = torch.zeros((batch_size, num_res), device=device)
        mask2.scatter_(dim=1, index=masked_sorted_max_logprobs_idcs, src=torch.ones((batch_size, num_res), device=device))
        unmask_zero_row = (number_to_unmask == 0).view(-1, 1).repeat(1, num_res).float()
        mask2 = mask2 * (1 - unmask_zero_row)
        aatypes_t = aatypes_t * (1 - mask2) + unmasked_samples * mask2

        # re-mask
        u = torch.rand(batch_size, num_res, device=self._device)
        re_mask_mask = (u < d_t * self._aatypes_cfg.noise).float()
        aatypes_t = aatypes_t * (1 - re_mask_mask) + du.MASK_TOKEN_INDEX * re_mask_mask

        return aatypes_t

    def _get_sigma_t(self, t):
        t_safe = torch.clamp(t, min=1e-5, max=1.0-1e-5)
        return self.noise_scale * torch.sqrt((1 - t_safe) / t_safe)
    
    def _trans_euler_step_sde(self, d_t, t, trans_1, trans_t):
        assert d_t >= 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)

        if self.noise_scale == 0:
            return trans_t + trans_vf * d_t
        
        sigma_t = self._get_sigma_t(t)

        t_safe = torch.clamp(t, max=1.0 - 1e-5) # 防止除以 0
        score = - (trans_t - t * trans_vf) / (1 - t_safe)
        drift_correction = 0.5 * (sigma_t ** 2) * score
        noise = torch.randn_like(trans_t) * sigma_t * (d_t ** 0.5)

        return trans_t + (trans_vf + drift_correction) * d_t + noise

    def _rots_euler_step_sde(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        
        rotmats_next = so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)
        
        return rotmats_next

    def _aatypes_euler_step_purity_sde(self, d_t, t, logits_1, aatypes_t):
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        assert S == 21
        assert self._aatypes_cfg.interpolant_type == "masking"
        device = logits_1.device

        logits_1_wo_mask = logits_1[:, :, 0:-1] # (B, D, S-1)
        pt_x1_probs = F.softmax(logits_1_wo_mask / self._aatypes_cfg.temp, dim=-1) # (B, D, S-1)
        # step_probs = (d_t * pt_x1_probs * (1/(1-t))).clamp(max=1) # (B, D, S-1)
        max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0] # (B, D)
        # bias so that only currently masked positions get chosen to be unmasked
        max_logprob = max_logprob - (aatypes_t != du.MASK_TOKEN_INDEX).float() * 1e9
        sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True) # (B, D)

        unmask_probs = (d_t * ( (1 + self._aatypes_cfg.noise * t) / (1-t)).to(device)).clamp(max=1) # scalar

        number_to_unmask = torch.binomial(count=torch.count_nonzero(aatypes_t == du.MASK_TOKEN_INDEX, dim=-1).float(),
                                          prob=unmask_probs)
        unmasked_samples = torch.multinomial(pt_x1_probs.view(-1, S-1), num_samples=1).view(batch_size, num_res)

        # Vectorized version of:
        # for b in range(B):
        #     for d in range(D):
        #         if d < number_to_unmask[b]:
        #             aatypes_t[b, sorted_max_logprobs_idcs[b, d]] = unmasked_samples[b, sorted_max_logprobs_idcs[b, d]]

        D_grid = torch.arange(num_res, device=device).view(1, -1).repeat(batch_size, 1)
        mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
        inital_val_max_logprob_idcs = sorted_max_logprobs_idcs[:, 0].view(-1, 1).repeat(1, num_res)
        masked_sorted_max_logprobs_idcs = (mask1 * sorted_max_logprobs_idcs + (1-mask1) * inital_val_max_logprob_idcs).long()
        mask2 = torch.zeros((batch_size, num_res), device=device)
        mask2.scatter_(dim=1, index=masked_sorted_max_logprobs_idcs, src=torch.ones((batch_size, num_res), device=device))
        unmask_zero_row = (number_to_unmask == 0).view(-1, 1).repeat(1, num_res).float()
        mask2 = mask2 * (1 - unmask_zero_row)
        aatypes_t = aatypes_t * (1 - mask2) + unmasked_samples * mask2

        # re-mask
        u = torch.rand(batch_size, num_res, device=self._device)
        re_mask_mask = (u < d_t * self._aatypes_cfg.noise).float()
        aatypes_t = aatypes_t * (1 - re_mask_mask) + du.MASK_TOKEN_INDEX * re_mask_mask

        return aatypes_t
        
    def calculate_metrics(
            self, 
            position, 
            aatypes, 
            target_position, 
            target_aatypes, 
            target_seg_lens, 
            binder_seq_len, 
            hotspot_str, 
            sample_dir, 
            sample_path_prefix):
        with jax.default_device(jax.devices()[dist.get_rank()]):
            # save generated binder sequence
            pred_fasta = fasta.FastaFile()
            pred_fasta['pred_seq_1'] = "".join([restypes_with_x[x] for x in aatypes])
            pred_fasta_path = os.path.join(sample_dir, sample_path_prefix + f'generated_binder.fasta')
            pred_fasta.write(pred_fasta_path)

            # save generated binder structure merged with all segs
            prot_pos_all_segs = []
            aatype_all_segs = []
            chain_splits_all_segs = []
            chain_ids_all_segs = []
            segs_total_len = 0
            for chain_id, target_seg_len in enumerate(target_seg_lens):
                prot_pos_all_segs.append(
                    target_position[segs_total_len:segs_total_len + target_seg_len].cpu().detach().numpy())
                aatype_all_segs.append(
                    target_aatypes[segs_total_len:segs_total_len + target_seg_len].cpu().detach().numpy())
                chain_splits_all_segs.append(target_seg_len)
                chain_ids_all_segs.append(self.alphabet[chain_id])
                segs_total_len += target_seg_len
            prot_pos_all_segs.append(position[-binder_seq_len:].cpu().detach().numpy())
            aatype_all_segs.append(aatypes[-binder_seq_len:].cpu().detach().numpy())
            chain_splits_all_segs.append(binder_seq_len)
            chain_ids_all_segs.append(self.alphabet[chain_id+1])
            pdb_path = au.write_prot_to_pdb_multi_chain(
                prot_pos=np.concatenate(prot_pos_all_segs, axis=0),
                file_path=os.path.join(sample_dir, sample_path_prefix + f'_merged_with_target_all_segs.pdb'),
                aatype=np.concatenate(aatype_all_segs, axis=0),
                no_indexing=True,
                chain_splits=chain_splits_all_segs,
                chain_ids=chain_ids_all_segs,
            )

            # get hotspot for metric calculation
            structure = self.parser.get_structure('struct', pdb_path)
            num_chains = len(list(structure[0].get_chains()))
            chain_offsets = {}
            current_offset = 0
            for chain in structure[0]:
                chain_offsets[chain.id] = current_offset
                current_offset += len(chain)
            new_hotspot_list = []
            for seg in hotspot_str.split(','):
                seg = seg.strip()
                if not seg:
                    continue
                
                c_id = seg[0]
                try:
                    local_pos = int(seg[1:])
                    
                    if c_id in chain_offsets:
                        global_pos = chain_offsets[c_id] + local_pos
                        new_hotspot_list.append(f"{c_id}{global_pos}")
                    else:
                        print(f"Warning: Chain {c_id} found in hotspot but not in PDB structure.")
                        new_hotspot_list.append(seg)
                except ValueError:
                    print(f"Warning: Could not parse hotspot segment '{seg}'")
                    new_hotspot_list.append(seg)
            new_hotspot_str = ",".join(new_hotspot_list)

            binder_seq = "".join([restypes_with_x[x] for x in aatypes])
            result = calculate_all_metrics(
                model=self.metric_model,
                binder_seq=binder_seq,
                gt_target_binder_pdb=pdb_path,
                hotspot=new_hotspot_str,
                target_chain=','.join([chr(65 + i) for i in range(num_chains - 1)]),
                binder_chain=','.join([chr(65 + num_chains - 1)]),
                num_recycles=0,
            )
            os.makedirs(os.path.join(sample_dir, 'AF2_Multimer_pred'), exist_ok=True)
            predicted_pdb_path = os.path.join(sample_dir, 'AF2_Multimer_pred', sample_path_prefix + 'predicted.pdb')
            if os.path.exists(predicted_pdb_path):
                os.remove(predicted_pdb_path)
            if hasattr(self.metric_model, "_tmp") and isinstance(self.metric_model._tmp, dict):
                best_pdb = self.metric_model._tmp.get("best", {}).get("aux", {}).get("pdb", None)
                if best_pdb is not None:
                    with open(predicted_pdb_path, "w", encoding="utf-8") as handle:
                        handle.write(best_pdb)
        
        return result

    def beam_search_sort(self, result_list):
        if len(result_list) == 1:
            return 0
        
        df = pd.DataFrame(result_list)

        w_ipae = 0.5
        w_plddt = 0.3
        w_scrmsd = 0.2

        df['norm_ipae'] = (df['ipae'].max() - df['ipae']) / (df['ipae'].max() - df['ipae'].min())
        df['norm_plddt'] = (df['binder_plddt'] - df['binder_plddt'].min()) / (df['binder_plddt'].max() - df['binder_plddt'].min())
        df['norm_scrmsd'] = (df['binder_scrmsd'].max() - df['binder_scrmsd']) / (df['binder_scrmsd'].max() - df['binder_scrmsd'].min())

        df['score'] = (df['norm_ipae'] * w_ipae) + \
                      (df['norm_plddt'] * w_plddt) + \
                      (df['norm_scrmsd'] * w_scrmsd)
        
        return df['score'].idxmax()
    
    def init_metric_model(self):
        if self.metric_model is None:
            with jax.default_device(jax.devices()[dist.get_rank()]):
                self.metric_model = mk_afdesign_model(**self.metric_model_kwargs)

    def sample(
            self,
            num_batch,
            num_res,
            model,
            num_timesteps=None,
            trans_0=None,
            rotmats_0=None,
            aatypes_0=None,
            trans_1=None,
            rotmats_1=None,
            aatypes_1=None,
            res_mask=None,
            diffuse_mask=None,
            chain_idx=None,
            res_idx=None,
            target_seq_len=None,
            binder_seq_len=None,
            hotspot=None,
            sample_dir=None,
            target_seg_lens=None,
            hotspot_str=None,
        ):
        if self.beam_search:
            self.init_metric_model()

        required_params = [trans_1, rotmats_1, aatypes_1, diffuse_mask, res_idx, chain_idx, target_seq_len, binder_seq_len]
        if not all(param is not None for param in required_params):
            raise ValueError(f'Parameters {required_params} must be provided for MSDesign sampling.')

        # res_mask = diffuse_mask.clone()
        for i, num_res_i in enumerate(num_res):
            res_mask[i][-num_res_i:] = 1.0

        # Set-up initial prior samples
        if trans_0 is None or rotmats_0 is None or aatypes_0 is None:
            trans_0 = trans_1.clone()
            rotmats_0 = rotmats_1.clone()
            aatypes_0 = aatypes_1.clone()
            for i, num_res_i in enumerate(num_res):
                # Set aatypes noise
                if self._aatypes_cfg.interpolant_type == "masking":
                    binder_aatypes_0_i = _masked_categorical(1, num_res_i, self._device)
                elif self._aatypes_cfg.interpolant_type == "uniform":
                    binder_aatypes_0_i = torch.randint_like(res_mask[i][-num_res_i:], low=0, high=self.num_tokens)
                else:
                    raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")
                aatypes_0[i][-num_res_i:] = binder_aatypes_0_i

                # Set trans and rotmats noise
                binder_trans_0_i = _centered_gaussian(1, num_res_i, self._device) * du.NM_TO_ANG_SCALE
                binder_rotmats_0_i = _uniform_so3(1, num_res_i, self._device)
                trans_0[i][-num_res_i:] = binder_trans_0_i
                rotmats_0[i][-num_res_i:] = binder_rotmats_0_i

        logits_1 = torch.nn.functional.one_hot(
            aatypes_1,
            num_classes=self.num_tokens
        ).float()

        trans_sc = torch.cat([
            torch.cat([trans_1[i:i+1, :-num_res_i], torch.zeros(1, num_res_i, 3, device=self._device)], dim=1) for i, num_res_i in enumerate(num_res)
        ], dim=0)
        aatypes_sc = torch.cat([
            torch.cat([logits_1[i:i+1, :-num_res_i], torch.zeros(1, num_res_i, self.num_tokens, device=self._device)], dim=1) for i, num_res_i in enumerate(num_res)
        ], dim=0)

        batch = {
            'res_mask': res_mask,
            'diffuse_mask': diffuse_mask,
            'chain_idx': chain_idx,
            'res_idx': res_idx,
            'trans_sc': trans_sc,
            'aatypes_sc': aatypes_sc,
            'target_seq_len': target_seq_len,
            'binder_seq_len': binder_seq_len,
            'hotspot': hotspot,
        }

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        frames_to_atom37 = lambda x,y: all_atom.atom37_from_trans_rot(x, y, None).detach().cpu()
        trans_t_1, rotmats_t_1, aatypes_t_1 = trans_0, rotmats_0, aatypes_0
        prot_traj = [[(frames_to_atom37(trans_t_1[i:i+1, -num_res_i:], rotmats_t_1[i:i+1, -num_res_i:]), aatypes_0[i:i+1, -num_res_i:].detach().cpu()) for i, num_res_i in enumerate(num_res)]] 
        clean_traj = []

        splited_ts = [ts[1:][max(0, i - self.beam_search_freq) : i] for i in range(len(ts[1:]), 0, -self.beam_search_freq)]
        splited_ts = splited_ts[::-1]

        start_trans_t_1 = trans_t_1
        start_rotmats_t_1 = rotmats_t_1
        start_aatypes_t_1 = aatypes_t_1
        start_t_1 = t_1
        for time_split_id, time_split in enumerate(splited_ts):
            candidate_trans_t_1 = {}
            candidate_rotmats_t_1 = {}
            candidate_aatypes_t_1 = {}
            candidate_prot_traj = {
                str(i): [] for i in range(self.beam_search_split_num)
            }
            candidate_clean_traj = {
                str(i): [] for i in range(self.beam_search_split_num)
            }
            for beam_search_i in range(self.beam_search_split_num):
                for t_2 in time_split:
                    if t_2 == time_split[0]:
                        trans_t_1 = start_trans_t_1
                        rotmats_t_1 = start_rotmats_t_1
                        aatypes_t_1 = start_aatypes_t_1
                        t_1 = start_t_1

                    # Run model.
                    batch['trans_t'] = torch.cat([
                        torch.cat([trans_1[i:i+1, :-num_res_i], trans_t_1[i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
                    ], dim=0)
                    batch['rotmats_t'] = torch.cat([
                        torch.cat([rotmats_1[i:i+1, :-num_res_i], rotmats_t_1[i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
                    ], dim=0)
                    batch['aatypes_t'] = torch.cat([
                        torch.cat([aatypes_1[i:i+1, :-num_res_i], aatypes_t_1[i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
                    ], dim=0)

                    t = torch.ones((num_batch, 1), device=self._device) * t_1
                    
                    if self._cfg.provide_kappa:
                        batch['binder_so3_t'] = self.rot_sample_kappa(t)
                    else:
                        batch['binder_so3_t'] = t
                    batch['binder_r3_t'] = t
                    batch['binder_cat_t'] = t
                    batch['target_so3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batch['binder_so3_t'])
                    batch['target_r3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batch['binder_r3_t'])
                    batch['target_cat_t'] = (1 - self._cfg.min_t) * torch.ones_like(batch['binder_cat_t'])

                    d_t = t_2 - t_1


                    with torch.no_grad():
                        model_out = model(batch)

                    # Process model output.
                    pred_trans_1 = model_out['pred_trans']
                    pred_rotmats_1 = model_out['pred_rotmats']
                    pred_aatypes_1 = model_out['pred_aatypes']
                    pred_logits_1 = model_out['pred_logits']
                    candidate_clean_traj[str(beam_search_i)].append([(frames_to_atom37(pred_trans_1[i:i+1, -num_res_i:], pred_rotmats_1[i:i+1, -num_res_i:]), pred_aatypes_1[i:i+1, -num_res_i:].detach().cpu()) for i, num_res_i in enumerate(num_res)])
                    
                    if self._cfg.self_condition:
                        batch['trans_sc'] = _trans_diffuse_mask(
                            pred_trans_1, trans_1, diffuse_mask)
                        batch['aatypes_sc'] = _trans_diffuse_mask(
                                pred_logits_1, logits_1, diffuse_mask)

                    # Take reverse step            
                    trans_t_2 = self._trans_euler_step_sde(
                        d_t, t_1, pred_trans_1, trans_t_1)
                    rotmats_t_2 = self._rots_euler_step_sde(
                        d_t, t_1, pred_rotmats_1, rotmats_t_1)

                    if self._aatypes_cfg.do_purity:
                        aatypes_t_2 = self._aatypes_euler_step_purity_sde(d_t, t_1, pred_logits_1, aatypes_t_1)
                    else:
                        raise ValueError('do_purity should be True')
                        aatypes_t_2 = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1)

                    trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
                    rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)
                    aatypes_t_2 = _aatypes_diffuse_mask(aatypes_t_2, aatypes_1, diffuse_mask)
                    trans_t_1, rotmats_t_1, aatypes_t_1 = trans_t_2, rotmats_t_2, aatypes_t_2
                    candidate_prot_traj[str(beam_search_i)].append([(frames_to_atom37(trans_t_2[i:i+1, -num_res_i:], rotmats_t_2[i:i+1, -num_res_i:]), aatypes_t_2[i:i+1, -num_res_i:].detach().cpu()) for i, num_res_i in enumerate(num_res)])

                    t_1 = t_2

                    candidate_trans_t_1[str(beam_search_i)] = trans_t_1
                    candidate_rotmats_t_1[str(beam_search_i)] = rotmats_t_1
                    candidate_aatypes_t_1[str(beam_search_i)] = aatypes_t_1

            # metrics calculation
            # assume batch size = 1
            assert sample_dir is not None
            assert target_seg_lens is not None
            assert hotspot_str is not None
            os.makedirs(sample_dir, exist_ok=True)
            result_list = []
            for idx, (k, v) in enumerate(candidate_clean_traj.items()):
                position = v[-1][0][0][0]
                aatypes = v[-1][0][1][0].int()
                target_position = all_atom.atom37_from_trans_rot(trans_1, rotmats_1)[0]
                target_aatypes = aatypes_1[0]
                result_i = self.calculate_metrics(
                    position=position,
                    aatypes=aatypes,
                    target_position=target_position,
                    target_aatypes=target_aatypes,
                    target_seg_lens=target_seg_lens[0].tolist(),
                    binder_seq_len=binder_seq_len[0],
                    hotspot_str=hotspot_str[0],
                    sample_dir=sample_dir,
                    sample_path_prefix=f'beam_search_{idx}_time_split_{time_split_id}_',
                )
                result_list.append(result_i)

            best_beam_search_id = self.beam_search_sort(result_list)

            # reset trans_t_1, rotmats_t_1, aatypes_t_1
            trans_t_1 = start_trans_t_1 = candidate_trans_t_1[str(best_beam_search_id)]
            rotmats_t_1 = start_rotmats_t_1 = candidate_rotmats_t_1[str(best_beam_search_id)]
            aatypes_t_1 = start_aatypes_t_1 = candidate_aatypes_t_1[str(best_beam_search_id)]
            start_t_1 = t_1

            # update traj
            prot_traj = prot_traj + candidate_prot_traj[str(best_beam_search_id)]
            clean_traj = clean_traj + candidate_clean_traj[str(best_beam_search_id)]

        t_1 = ts[-1]

        batch['trans_t'] = torch.cat([
            torch.cat([trans_1[i:i+1, :-num_res_i], trans_t_1[i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
        ], dim=0)
        batch['rotmats_t'] = torch.cat([
            torch.cat([rotmats_1[i:i+1, :-num_res_i], rotmats_t_1[i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
        ], dim=0)
        batch['aatypes_t'] = torch.cat([
            torch.cat([aatypes_1[i:i+1, :-num_res_i], aatypes_t_1[i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
        ], dim=0)

        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        pred_aatypes_1 = model_out['pred_aatypes']
        
        final_traj = [(frames_to_atom37(pred_trans_1[i:i+1, -num_res_i:], pred_rotmats_1[i:i+1, -num_res_i:]), pred_aatypes_1[i:i+1, -num_res_i:].detach().cpu()) for i, num_res_i in enumerate(num_res)]
        clean_traj.append(final_traj)
        prot_traj.append(final_traj)
        return prot_traj, clean_traj
    
    def calculate_metrics_multi_target(
            self, 
            position, 
            aatypes, 
            target_position, 
            target_aatypes, 
            target_seq_len, 
            binder_seq_len, 
            hotspot_str, 
            sample_dir, 
            sample_path_prefix, 
            target_id):
        with jax.default_device(jax.devices()[dist.get_rank()]):
            # save generated binder sequence
            pred_fasta = fasta.FastaFile()
            pred_fasta['pred_seq_1'] = "".join([restypes_with_x[x] for x in aatypes])
            pred_fasta_path = os.path.join(sample_dir, sample_path_prefix + f'generated_binder.fasta')
            pred_fasta.write(pred_fasta_path)

            # merge binder with target
            pdb_path = au.write_prot_to_pdb_multi_chain(
                prot_pos=np.concatenate([
                    target_position[:target_seq_len].cpu().detach().numpy(),
                    position[-binder_seq_len:].cpu().detach().numpy(),
                ], axis=0),
                file_path=os.path.join(sample_dir, sample_path_prefix + f'_merged_with_target_{target_id}.pdb'),
                aatype=np.concatenate([
                    target_aatypes[:target_seq_len].cpu().detach().numpy(),
                    aatypes[-binder_seq_len:].cpu().detach().numpy(),
                ], axis=0),
                no_indexing=True,
                chain_splits=[target_seq_len, binder_seq_len],
                chain_ids=['A', 'B'],
            )

            binder_seq = "".join([restypes_with_x[x] for x in aatypes])
            result = calculate_all_metrics(
                model=self.metric_model,
                binder_seq=binder_seq,
                gt_target_binder_pdb=pdb_path,
                hotspot=hotspot_str,
                target_chain='A',
                binder_chain='B',
                num_recycles=0,
            )
            os.makedirs(os.path.join(sample_dir, 'AF2_Multimer_pred'), exist_ok=True)
            predicted_pdb_path = os.path.join(sample_dir, 'AF2_Multimer_pred', sample_path_prefix + 'predicted.pdb')
            if os.path.exists(predicted_pdb_path):
                os.remove(predicted_pdb_path)
            if hasattr(self.metric_model, "_tmp") and isinstance(self.metric_model._tmp, dict):
                best_pdb = self.metric_model._tmp.get("best", {}).get("aux", {}).get("pdb", None)
                if best_pdb is not None:
                    with open(predicted_pdb_path, "w", encoding="utf-8") as handle:
                        handle.write(best_pdb)
        
        return result
    
    def beam_search_sort_multi_target(result_list):
        metrics = ["ipae", "binder_plddt", "binder_scrmsd"]

        rows = []
        for candidate_idx, result_list_i in enumerate(result_list):
            row = {"candidate_idx": candidate_idx}
            for j, d in enumerate(result_list_i):
                for m in metrics:
                    row[f"{m}_{j}"] = d[m]
            rows.append(row)

        df = pd.DataFrame(rows)

        w_ipae = 0.5
        w_plddt = 0.3
        w_scrmsd = 0.2

        for j in range(len(result_list[0])):
            df[f'norm_ipae_{j}'] = (df[f'ipae_{j}'].max() - df[f'ipae_{j}']) / (df[f'ipae_{j}'].max() - df[f'ipae_{j}'].min())
            df[f'norm_plddt_{j}'] = (df[f'binder_plddt_{j}'] - df[f'binder_plddt_{j}'].min()) / (df[f'binder_plddt_{j}'].max() - df[f'binder_plddt_{j}'].min())
            df[f'norm_scrmsd_{j}'] = (df[f'binder_scrmsd_{j}'].max() - df[f'binder_scrmsd_{j}']) / (df[f'binder_scrmsd_{j}'].max() - df[f'binder_scrmsd_{j}'].min())

        state_scores = []
        for j in range(len(result_list[0])):
            s_j = (
                df[f"norm_ipae_{j}"] * w_ipae +
                df[f"norm_plddt_{j}"] * w_plddt +
                df[f"norm_scrmsd_{j}"] * w_scrmsd
            )
            df[f"score_{j}"] = s_j
            state_scores.append(s_j)

        df["score"] = pd.concat(state_scores, axis=1).mean(axis=1)
        
        return df['score'].idxmax()

    def multi_target_sample(
            self,
            num_batch,
            num_res,
            model,
            num_timesteps=None,
            switch_steps=None,
            trans_0=None,
            rotmats_0=None,
            aatypes_0=None,
            trans_1=None,
            rotmats_1=None,
            aatypes_1=None,
            res_mask=None,
            diffuse_mask=None,
            chain_idx=None,
            res_idx=None,
            target_seq_len=None,
            binder_seq_len=None,
            hotspot=None,
            sample_dir=None,
            hotspot_str=None,
        ):
        if self.beam_search:
            self.init_metric_model()

        required_params = [trans_1, rotmats_1, aatypes_1, diffuse_mask, res_idx, chain_idx, target_seq_len, binder_seq_len]
        if not all(param is not None for param in required_params):
            raise ValueError(f'Parameters {required_params} must be provided for MSDesign sampling.')

        # res_mask = [[diffuse_mask_j.clone() for diffuse_mask_j in batch_diffuse_mask_i] for batch_diffuse_mask_i in diffuse_mask]
        # Set diffuse_mask, res_mask, chain_idx, res_idx
        for i, num_res_i in enumerate(num_res):
            for j, target_seq_len_j in enumerate(target_seq_len[i]):
                res_mask[i][j] = torch.cat([
                    res_mask[i][j][:, :target_seq_len_j],
                    torch.ones(1, num_res_i, device=self._device, dtype=res_mask[i][j].dtype)
                ], dim=1)
                diffuse_mask[i][j] = torch.cat([
                    diffuse_mask[i][j][:, :target_seq_len_j],
                    torch.ones(1, num_res_i, device=self._device, dtype=diffuse_mask[i][j].dtype)
                ], dim=1)
                hotspot[i][j] = torch.cat([
                    hotspot[i][j][:, :target_seq_len_j],
                    torch.zeros(1, num_res_i, device=self._device, dtype=hotspot[i][j].dtype)
                ], dim=1)
                chain_idx[i][j] = torch.cat([
                    chain_idx[i][j][:, :target_seq_len_j],
                    torch.ones(1, num_res_i, device=self._device, dtype=chain_idx[i][j].dtype) * chain_idx[i][j][0, target_seq_len_j].item()
                ], dim=1)
                res_idx[i][j] = torch.cat([
                    res_idx[i][j][:, :target_seq_len_j],
                    torch.arange(num_res_i, device=self._device, dtype=res_idx[i][j].dtype).view(1, -1) + res_idx[i][j][0, target_seq_len_j].item()
                ], dim=1)

        # Set-up initial prior samples
        if trans_0 is None or rotmats_0 is None or aatypes_0 is None:
            trans_0 = [[trans_1_j.clone() for trans_1_j in batch_trans_1_i] for batch_trans_1_i in trans_1]
            rotmats_0 = [[rotmats_1_j.clone() for rotmats_1_j in batch_rotmats_1_i] for batch_rotmats_1_i in rotmats_1]
            aatypes_0 = [[aatypes_1_j.clone() for aatypes_1_j in batch_aatypes_1_i] for batch_aatypes_1_i in aatypes_1]
            for i, num_res_i in enumerate(num_res):
                # Set aatypes noise
                if self._aatypes_cfg.interpolant_type == "masking":
                    binder_aatypes_0_i = _masked_categorical(1, num_res_i, self._device)
                elif self._aatypes_cfg.interpolant_type == "uniform":
                    # TODO: here may be a risk (num_res_i may be larger than the number of residues in res_mask[i][j])
                    # but currently we only use "masking" strategy
                    binder_aatypes_0_i = torch.randint_like(res_mask[i][j][:, -num_res_i:], low=0, high=self.num_tokens)
                else:
                    raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")
                for j, target_seq_len_j in enumerate(target_seq_len[i]):
                    aatypes_0[i][j] = torch.cat([
                        aatypes_0[i][j][:, :target_seq_len_j],
                        binder_aatypes_0_i
                    ], dim=1)

                    # Set trans and rotmats noise
                    binder_trans_0_j = _centered_gaussian(1, num_res_i, self._device) * du.NM_TO_ANG_SCALE
                    binder_rotmats_0_j = _uniform_so3(1, num_res_i, self._device)
                    trans_0[i][j] = torch.cat([
                        trans_0[i][j][:, :target_seq_len_j],
                        binder_trans_0_j
                    ], dim=1)
                    rotmats_0[i][j] = torch.cat([
                        rotmats_0[i][j][:, :target_seq_len_j],
                        binder_rotmats_0_j
                    ], dim=1)


        logits_1 = [[torch.nn.functional.one_hot(aatypes_1_j, num_classes=self.num_tokens) for aatypes_1_j in batch_aatypes_1_i] for batch_aatypes_1_i in aatypes_1]

        trans_sc = []
        aatypes_sc = []
        for i, num_res_i in enumerate(num_res):
            trans_sc.append([])
            aatypes_sc.append([])
            for j, target_seq_len_j in enumerate(target_seq_len[i]):
                trans_sc[i].append(torch.cat([
                    trans_1[i][j][:, :target_seq_len_j],
                    torch.zeros(1, num_res_i, 3, device=self._device)
                ], dim=1))
                aatypes_sc[i].append(torch.cat([
                    logits_1[i][j][:, :target_seq_len_j],
                    torch.zeros(1, num_res_i, self.num_tokens, device=self._device)
                ], dim=1))

        # re-orgnize the batch, we assume the batch size is 1
        batches = []
        for j, target_seq_len_j in enumerate(target_seq_len[0]):
            batch_j = {}
            batch_j['res_mask'] = torch.cat([
                res_mask[i][j] for i, _ in enumerate(num_res)
            ], dim=0)
            batch_j['diffuse_mask'] = torch.cat([
                diffuse_mask[i][j] for i, _ in enumerate(num_res)
            ], dim=0)
            batch_j['chain_idx'] = torch.cat([
                chain_idx[i][j] for i, _ in enumerate(num_res)
            ], dim=0)
            batch_j['res_idx'] = torch.cat([
                res_idx[i][j] for i, _ in enumerate(num_res)
            ], dim=0)
            batch_j['trans_sc'] = torch.cat([
                trans_sc[i][j] for i, _ in enumerate(num_res)
            ], dim=0)
            batch_j['aatypes_sc'] = torch.cat([
                aatypes_sc[i][j] for i, _ in enumerate(num_res)
            ], dim=0)
            batch_j['target_seq_len'] = torch.cat([
                torch.as_tensor(target_seq_len[i][j], dtype=torch.long, device=batch_j['res_mask'].device).unsqueeze(0) 
                for i, _ in enumerate(num_res)
            ], dim=0)
            batch_j['binder_seq_len'] = torch.ones_like(batch_j['target_seq_len'])
            for i, num_res_i in enumerate(num_res):
                batch_j['binder_seq_len'][i] = batch_j['binder_seq_len'][i] * num_res_i
            batch_j['hotspot'] = torch.cat([
                hotspot[i][j] for i, _ in enumerate(num_res)
            ], dim=0)
            batch_j['trans_t_1'] = torch.cat([
                trans_0[i][j] for i, _ in enumerate(num_res)
            ], dim=0)
            batch_j['rotmats_t_1'] = torch.cat([
                rotmats_0[i][j] for i, _ in enumerate(num_res)
            ], dim=0)
            batch_j['aatypes_t_1'] = torch.cat([
                aatypes_0[i][j] for i, _ in enumerate(num_res)
            ], dim=0)
            batches.append(batch_j)
        
        reorgnized_trans_1 = [
            torch.cat([torch.cat([trans_1[i][j][:, :target_seq_len[i][j]], trans_0[i][j][:, -num_res[i]:]], dim=1) for i in range(len(num_res))], dim=0)
            for j in range(len(target_seq_len[0]))
        ]
        reorgnized_rotmats_1 = [
            torch.cat([torch.cat([rotmats_1[i][j][:, :target_seq_len[i][j]], rotmats_0[i][j][:, -num_res[i]:]], dim=1) for i in range(len(num_res))], dim=0)
            for j in range(len(target_seq_len[0]))
        ]
        reorgnized_aatypes_1 = [
            torch.cat([torch.cat([aatypes_1[i][j][:, :target_seq_len[i][j]], aatypes_0[i][j][:, -num_res[i]:]], dim=1) for i in range(len(num_res))], dim=0)
            for j in range(len(target_seq_len[0]))
        ]
        reorgnized_logits_1 = [
            torch.cat([torch.cat([logits_1[i][j][:, :target_seq_len[i][j]], torch.zeros(1, num_res[i], self.num_tokens, device=self._device)], dim=1) for i in range(len(num_res))], dim=0)
            for j in range(len(target_seq_len[0]))
        ]

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]
        for j in range(len(batches)):
            batches[j]['cur_diffusion_t_idx'] = 0
        if switch_steps is None:
            switch_steps = self._sample_cfg.switch_steps
        if num_timesteps % switch_steps != 0:
            raise ValueError(f'Number of timesteps {num_timesteps} must be divisible by switch_steps {switch_steps}.')

        frames_to_atom37 = lambda x,y: all_atom.atom37_from_trans_rot(x, y, None).detach().cpu()
        # trans_t_1, rotmats_t_1, aatypes_t_1 = [trans_0] * len(batches), [rotmats_0] * len(batches), [aatypes_0] * len(batches)
        prot_traj = [
            [
                (
                    {
                        str(j): frames_to_atom37(batches[j]['trans_t_1'][i:i+1, -num_res_i:], batches[j]['rotmats_t_1'][i:i+1, -num_res_i:]) 
                        for j in range(len(batches))
                    }, 
                    batches[0]['aatypes_t_1'][i:i+1, -num_res_i:].detach().cpu()
                ) 
                for i, num_res_i in enumerate(num_res)
            ]
        ] 
        clean_traj = []
        cur_traj_idx = 0
        next_traj_idx = (cur_traj_idx + 1) % len(batches)
        condition_t1 = t_1
        condition_t2 = ts[1]
        condition_t_idx = 0

        splited_ts = [ts[1:][max(0, i - self.beam_search_freq) : i] for i in range(len(ts[1:]), 0, -self.beam_search_freq)]
        splited_ts = splited_ts[::-1]
        all_t_idx = list(range(len(ts[1:])))
        splited_t_idx = [all_t_idx[max(0, i - self.beam_search_freq) : i] for i in range(len(all_t_idx), 0, -self.beam_search_freq)]
        splited_t_idx = splited_t_idx[::-1]

        start_trans_t_1 = [batches[j]['trans_t_1'] for j in range(len(batches))]
        start_rotmats_t_1 = [batches[j]['rotmats_t_1'] for j in range(len(batches))]
        start_aatypes_t_1 = [batches[j]['aatypes_t_1'] for j in range(len(batches))]
        start_t_1 = t_1
        start_pred_aatypes_1 = [batches[j]['aatypes_t_1'] for j in range(len(batches))]
        start_cur_traj_idx = cur_traj_idx
        start_next_traj_idx = next_traj_idx
        for time_split_id, (t_idx_split, time_split) in enumerate(zip(splited_t_idx, splited_ts)):
            candidate_trans_t_1 = {}
            candidate_rotmats_t_1 = {}
            candidate_aatypes_t_1 = {}
            candidate_prot_traj = {
                str(i): copy.deepcopy(prot_traj) for i in range(self.beam_search_split_num)
            }
            candidate_clean_traj = {
                str(i): copy.deepcopy(clean_traj) for i in range(self.beam_search_split_num)
            }
            for beam_search_i in range(self.beam_search_split_num):
                for j in range(len(batches)):
                    batches[j]['trans_t_1'] = start_trans_t_1[j]
                    batches[j]['rotmats_t_1'] = start_rotmats_t_1[j]
                    batches[j]['aatypes_t_1'] = start_aatypes_t_1[j]
                t_1 = start_t_1
                cur_traj_idx = start_cur_traj_idx
                next_traj_idx = start_next_traj_idx
                for t_idx, t_2 in zip(t_idx_split, time_split):

                    if batches[cur_traj_idx].get('pred_aatypes_1', None) is not None and t_2 == time_split[0] and t_idx % switch_steps == 0 and t_idx != 0:
                        if beam_search_i == 0:
                            start_pred_aatypes_1 = [batches[j]['pred_aatypes_1'] for j in range(len(batches))]
                        else:
                            for j in range(len(batches)):
                                batches[j]['pred_aatypes_1'] = start_pred_aatypes_1[j]

                    if t_idx % switch_steps == 0 and t_idx != 0:
                        condition_ts = ts[max(1, t_idx + 1 - switch_steps * (len(batches) - 1)):t_idx + 1]
                        condition_t1 = ts[max(0, t_idx - switch_steps * (len(batches) - 1))]
                        for condition_t_idx, condition_t2 in enumerate(condition_ts):
                            batches[next_traj_idx]['trans_t'] = torch.cat([
                                torch.cat([reorgnized_trans_1[next_traj_idx][i:i+1, :target_seq_len[i][next_traj_idx]], batches[next_traj_idx]['trans_t_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
                            ], dim=0)
                            batches[next_traj_idx]['rotmats_t'] = torch.cat([
                                torch.cat([reorgnized_rotmats_1[next_traj_idx][i:i+1, :target_seq_len[i][next_traj_idx]], batches[next_traj_idx]['rotmats_t_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
                            ], dim=0)
                            batches[next_traj_idx]['aatypes_t'] = torch.cat([
                                torch.cat([reorgnized_aatypes_1[next_traj_idx][i:i+1, :target_seq_len[i][next_traj_idx]], batches[cur_traj_idx]['pred_aatypes_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
                            ], dim=0)

                            t = torch.ones((num_batch, 1), device=self._device) * condition_t1

                            if self._cfg.provide_kappa:
                                batches[next_traj_idx]['binder_so3_t'] = self.rot_sample_kappa(t)
                            else:
                                batches[next_traj_idx]['binder_so3_t'] = t
                            batches[next_traj_idx]['binder_r3_t'] = t
                            batches[next_traj_idx]['binder_cat_t'] = (1 - self._cfg.min_t) * torch.ones((num_batch, 1), device=self._device)
                            batches[next_traj_idx]['target_so3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batches[next_traj_idx]['binder_so3_t'])
                            batches[next_traj_idx]['target_r3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batches[next_traj_idx]['binder_r3_t'])
                            batches[next_traj_idx]['target_cat_t'] = (1 - self._cfg.min_t) * torch.ones_like(batches[next_traj_idx]['binder_cat_t'])

                            condition_dt = condition_t2 - condition_t1

                            with torch.no_grad():
                                model_out = model(batches[next_traj_idx])

                            batches[next_traj_idx]['pred_trans_1'] = model_out['pred_trans']
                            batches[next_traj_idx]['pred_rotmats_1'] = model_out['pred_rotmats']
                            batches[next_traj_idx]['pred_aatypes_1'] = model_out['pred_aatypes']
                            batches[next_traj_idx]['pred_logits_1'] = model_out['pred_logits']
                            for i, num_res_i in enumerate(num_res):
                                candidate_clean_traj[str(beam_search_i)][max(0, t_idx - switch_steps * (len(batches) - 1)) + condition_t_idx][i][0][str(next_traj_idx)] = frames_to_atom37(batches[next_traj_idx]['pred_trans_1'][i:i+1, -num_res_i:], batches[next_traj_idx]['pred_rotmats_1'][i:i+1, -num_res_i:])

                            if self._cfg.self_condition:
                                batches[next_traj_idx]['trans_sc'] = _trans_diffuse_mask(
                                    batches[next_traj_idx]['pred_trans_1'], reorgnized_trans_1[next_traj_idx], batches[next_traj_idx]['diffuse_mask'])
                                batches[next_traj_idx]['aatypes_sc'] = _trans_diffuse_mask(
                                    batches[next_traj_idx]['pred_logits_1'], reorgnized_logits_1[next_traj_idx], batches[next_traj_idx]['diffuse_mask'])

                            # Take reverse step            
                            batches[next_traj_idx]['trans_t_2'] = self._trans_euler_step_sde(
                                condition_dt, condition_t1, batches[next_traj_idx]['pred_trans_1'], batches[next_traj_idx]['trans_t_1'])
                            batches[next_traj_idx]['rotmats_t_2'] = self._rots_euler_step_sde(
                                condition_dt, condition_t1, batches[next_traj_idx]['pred_rotmats_1'], batches[next_traj_idx]['rotmats_t_1'])

                            if self._aatypes_cfg.do_purity:
                                batches[next_traj_idx]['aatypes_t_2'] = self._aatypes_euler_step_purity_sde(condition_dt, condition_t1, batches[next_traj_idx]['pred_logits_1'], batches[next_traj_idx]['aatypes_t_1'])
                            else:
                                raise ValueError('do_purity should be True')
                                batches[next_traj_idx]['aatypes_t_2'] = self._aatypes_euler_step(condition_dt, condition_t1, batches[next_traj_idx]['pred_logits_1'], batches[next_traj_idx]['aatypes_t_1'])

                            batches[next_traj_idx]['trans_t_2'] = _trans_diffuse_mask(batches[next_traj_idx]['trans_t_2'], reorgnized_trans_1[next_traj_idx], batches[next_traj_idx]['diffuse_mask'])
                            batches[next_traj_idx]['rotmats_t_2'] = _rots_diffuse_mask(batches[next_traj_idx]['rotmats_t_2'], reorgnized_rotmats_1[next_traj_idx], batches[next_traj_idx]['diffuse_mask'])
                            batches[next_traj_idx]['aatypes_t_2'] = _aatypes_diffuse_mask(batches[next_traj_idx]['aatypes_t_2'], reorgnized_aatypes_1[next_traj_idx], batches[next_traj_idx]['diffuse_mask'])
                            batches[next_traj_idx]['trans_t_1'], batches[next_traj_idx]['rotmats_t_1'], batches[next_traj_idx]['aatypes_t_1'] = batches[next_traj_idx]['trans_t_2'], batches[next_traj_idx]['rotmats_t_2'], batches[next_traj_idx]['aatypes_t_2']
                            for i, num_res_i in enumerate(num_res):
                                candidate_prot_traj[str(beam_search_i)][max(0, t_idx - switch_steps * (len(batches) - 1)) + condition_t_idx + 1][i][0][str(next_traj_idx)] = frames_to_atom37(batches[next_traj_idx]['trans_t_2'][i:i+1, -num_res_i:], batches[next_traj_idx]['rotmats_t_2'][i:i+1, -num_res_i:])

                            condition_t1 = condition_t2
                            batches[next_traj_idx]['cur_diffusion_t_idx'] = max(0, t_idx - switch_steps * (len(batches) - 1)) + condition_t_idx + 1

                        cur_traj_idx = next_traj_idx
                        next_traj_idx = (cur_traj_idx + 1) % len(batches)

                    # Run model.
                    batches[cur_traj_idx]['trans_t'] = torch.cat([
                        torch.cat([reorgnized_trans_1[cur_traj_idx][i:i+1, :target_seq_len[i][cur_traj_idx]], batches[cur_traj_idx]['trans_t_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
                    ], dim=0)
                    batches[cur_traj_idx]['rotmats_t'] = torch.cat([
                        torch.cat([reorgnized_rotmats_1[cur_traj_idx][i:i+1, :target_seq_len[i][cur_traj_idx]], batches[cur_traj_idx]['rotmats_t_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
                    ], dim=0)
                    batches[cur_traj_idx]['aatypes_t'] = torch.cat([
                        torch.cat([reorgnized_aatypes_1[cur_traj_idx][i:i+1, :target_seq_len[i][cur_traj_idx]], batches[cur_traj_idx]['aatypes_t_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
                    ], dim=0)

                    t = torch.ones((num_batch, 1), device=self._device) * t_1
                    
                    if self._cfg.provide_kappa:
                        batches[cur_traj_idx]['binder_so3_t'] = self.rot_sample_kappa(t)
                    else:
                        batches[cur_traj_idx]['binder_so3_t'] = t
                    batches[cur_traj_idx]['binder_r3_t'] = t
                    batches[cur_traj_idx]['binder_cat_t'] = t
                    batches[cur_traj_idx]['target_so3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batches[cur_traj_idx]['binder_so3_t'])
                    batches[cur_traj_idx]['target_r3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batches[cur_traj_idx]['binder_r3_t'])
                    batches[cur_traj_idx]['target_cat_t'] = (1 - self._cfg.min_t) * torch.ones_like(batches[cur_traj_idx]['binder_cat_t'])

                    d_t = t_2 - t_1


                    with torch.no_grad():
                        model_out = model(batches[cur_traj_idx])

                    # Process model output.
                    batches[cur_traj_idx]['pred_trans_1'] = model_out['pred_trans']
                    batches[cur_traj_idx]['pred_rotmats_1'] = model_out['pred_rotmats']
                    batches[cur_traj_idx]['pred_aatypes_1'] = model_out['pred_aatypes']
                    batches[cur_traj_idx]['pred_logits_1'] = model_out['pred_logits']
                    candidate_clean_traj[str(beam_search_i)].append([({}, batches[cur_traj_idx]['pred_aatypes_1'][i:i+1, -num_res_i:].detach().cpu()) for i, num_res_i in enumerate(num_res)])
                    for i, num_res_i in enumerate(num_res):
                        candidate_clean_traj[str(beam_search_i)][-1][i][0][str(cur_traj_idx)] = frames_to_atom37(batches[cur_traj_idx]['pred_trans_1'][i:i+1, -num_res_i:], batches[cur_traj_idx]['pred_rotmats_1'][i:i+1, -num_res_i:])

                    if self._cfg.self_condition:
                        batches[cur_traj_idx]['trans_sc'] = _trans_diffuse_mask(
                            batches[cur_traj_idx]['pred_trans_1'], reorgnized_trans_1[cur_traj_idx], batches[cur_traj_idx]['diffuse_mask'])
                        batches[cur_traj_idx]['aatypes_sc'] = _trans_diffuse_mask(
                            batches[cur_traj_idx]['pred_logits_1'], reorgnized_logits_1[cur_traj_idx], batches[cur_traj_idx]['diffuse_mask'])
                    # Take reverse step            
                    batches[cur_traj_idx]['trans_t_2'] = self._trans_euler_step_sde(
                        d_t, t_1, batches[cur_traj_idx]['pred_trans_1'], batches[cur_traj_idx]['trans_t_1'])
                    batches[cur_traj_idx]['rotmats_t_2'] = self._rots_euler_step_sde(
                        d_t, t_1, batches[cur_traj_idx]['pred_rotmats_1'], batches[cur_traj_idx]['rotmats_t_1'])

                    if self._aatypes_cfg.do_purity:
                        batches[cur_traj_idx]['aatypes_t_2'] = self._aatypes_euler_step_purity_sde(d_t, t_1, batches[cur_traj_idx]['pred_logits_1'], batches[cur_traj_idx]['aatypes_t_1'])
                    else:
                        raise ValueError('do_purity should be True')
                        batches[cur_traj_idx]['aatypes_t_2'] = self._aatypes_euler_step(d_t, t_1, batches[cur_traj_idx]['pred_logits_1'], batches[cur_traj_idx]['aatypes_t_1'])

                    batches[cur_traj_idx]['trans_t_2'] = _trans_diffuse_mask(batches[cur_traj_idx]['trans_t_2'], reorgnized_trans_1[cur_traj_idx], batches[cur_traj_idx]['diffuse_mask'])
                    batches[cur_traj_idx]['rotmats_t_2'] = _rots_diffuse_mask(batches[cur_traj_idx]['rotmats_t_2'], reorgnized_rotmats_1[cur_traj_idx], batches[cur_traj_idx]['diffuse_mask'])
                    batches[cur_traj_idx]['aatypes_t_2'] = _aatypes_diffuse_mask(batches[cur_traj_idx]['aatypes_t_2'], reorgnized_aatypes_1[cur_traj_idx], batches[cur_traj_idx]['diffuse_mask'])
                    batches[cur_traj_idx]['trans_t_1'], batches[cur_traj_idx]['rotmats_t_1'], batches[cur_traj_idx]['aatypes_t_1'] = batches[cur_traj_idx]['trans_t_2'], batches[cur_traj_idx]['rotmats_t_2'], batches[cur_traj_idx]['aatypes_t_2']
                    candidate_prot_traj[str(beam_search_i)].append([({}, batches[cur_traj_idx]['aatypes_t_2'][i:i+1, -num_res_i:].detach().cpu()) for i, num_res_i in enumerate(num_res)])
                    for i, num_res_i in enumerate(num_res):
                        candidate_prot_traj[str(beam_search_i)][-1][i][0][str(cur_traj_idx)] = frames_to_atom37(batches[cur_traj_idx]['trans_t_2'][i:i+1, -num_res_i:], batches[cur_traj_idx]['rotmats_t_2'][i:i+1, -num_res_i:])
                    
                    t_1 = t_2
                    batches[cur_traj_idx]['cur_diffusion_t_idx'] = t_idx + 1

                    candidate_trans_t_1[str(beam_search_i)] = [batches[j]['trans_t_1'] for j in range(len(batches))]
                    candidate_rotmats_t_1[str(beam_search_i)] = [batches[j]['rotmats_t_1'] for j in range(len(batches))]
                    candidate_aatypes_t_1[str(beam_search_i)] = [batches[j]['aatypes_t_1'] for j in range(len(batches))]
            
            # metrics calculation
            # assume batch size = 1
            assert sample_dir is not None
            assert hotspot_str is not None
            os.makedirs(sample_dir, exist_ok=True)
            result_list = []
            for idx, (k, v) in enumerate(candidate_clean_traj.items()):
                aatypes = v[-1][0][1][0].int()
                position = v[-1][0][0][str(cur_traj_idx)][0]
                target_position = all_atom.atom37_from_trans_rot(reorgnized_trans_1[cur_traj_idx], reorgnized_rotmats_1[cur_traj_idx])[0]
                target_aatypes = reorgnized_aatypes_1[cur_traj_idx][0].int()
                result_i = self.calculate_metrics_multi_target(
                    position=position,
                    aatypes=aatypes,
                    target_position=target_position,
                    target_aatypes=target_aatypes,
                    target_seq_len=target_seq_len[0][cur_traj_idx],
                    binder_seq_len=num_res[0],
                    hotspot_str=hotspot_str[0][cur_traj_idx],
                    sample_dir=sample_dir,
                    sample_path_prefix=f'beam_search_{idx}_time_split_{time_split_id}_target_{j}',
                    target_id=j,
                )
                result_list.append(result_i)
            
            best_beam_search_id = self.beam_search_sort(result_list)

            # reset trans_t_1, rotmats_t_1, aatypes_t_1
            for j in range(len(batches)):
                batches[j]['trans_t_1'] = start_trans_t_1[j] = candidate_trans_t_1[str(best_beam_search_id)][j]
                batches[j]['rotmats_t_1'] = start_rotmats_t_1[j] = candidate_rotmats_t_1[str(best_beam_search_id)][j]
                batches[j]['aatypes_t_1'] = start_aatypes_t_1[j] = candidate_aatypes_t_1[str(best_beam_search_id)][j]
            start_t_1 = t_1
            start_cur_traj_idx = cur_traj_idx
            start_next_traj_idx = next_traj_idx

            # update traj
            prot_traj = candidate_prot_traj[str(best_beam_search_id)]
            clean_traj = candidate_clean_traj[str(best_beam_search_id)]

        # We only integrated to min_t, so need to make a final step
        t_idx += 1
        t_1 = ts[-1]

        batches[cur_traj_idx]['trans_t'] = torch.cat([
            torch.cat([reorgnized_trans_1[cur_traj_idx][i:i+1, :target_seq_len[i][cur_traj_idx]], batches[cur_traj_idx]['trans_t_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
        ], dim=0)
        batches[cur_traj_idx]['rotmats_t'] = torch.cat([
            torch.cat([reorgnized_rotmats_1[cur_traj_idx][i:i+1, :target_seq_len[i][cur_traj_idx]], batches[cur_traj_idx]['rotmats_t_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
        ], dim=0)
        batches[cur_traj_idx]['aatypes_t'] = torch.cat([
            torch.cat([reorgnized_aatypes_1[cur_traj_idx][i:i+1, :target_seq_len[i][cur_traj_idx]], batches[cur_traj_idx]['aatypes_t_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
        ], dim=0)

        with torch.no_grad():
            model_out = model(batches[cur_traj_idx])
        batches[cur_traj_idx]['pred_trans_1'] = model_out['pred_trans']
        batches[cur_traj_idx]['pred_rotmats_1'] = model_out['pred_rotmats']
        batches[cur_traj_idx]['pred_aatypes_1'] = model_out['pred_aatypes']
        
        clean_traj.append([({}, batches[cur_traj_idx]['pred_aatypes_1'][i:i+1, -num_res_i:].detach().cpu()) for i, num_res_i in enumerate(num_res)])
        for i, num_res_i in enumerate(num_res):
            clean_traj[-1][i][0][str(cur_traj_idx)] = frames_to_atom37(batches[cur_traj_idx]['pred_trans_1'][i:i+1, -num_res_i:], batches[cur_traj_idx]['pred_rotmats_1'][i:i+1, -num_res_i:])
        prot_traj.append([({}, batches[cur_traj_idx]['pred_aatypes_1'][i:i+1, -num_res_i:].detach().cpu()) for i, num_res_i in enumerate(num_res)])
        for i, num_res_i in enumerate(num_res):
            prot_traj[-1][i][0][str(cur_traj_idx)] = frames_to_atom37(batches[cur_traj_idx]['pred_trans_1'][i:i+1, -num_res_i:], batches[cur_traj_idx]['pred_rotmats_1'][i:i+1, -num_res_i:])

        # finish other states
        t_idx += 1
        for j in range(1, len(batches)):
            last_traj_idx = (cur_traj_idx + j) % len(batches)
            last_ts = ts[batches[last_traj_idx]['cur_diffusion_t_idx']+1:]
            last_t1 = ts[batches[last_traj_idx]['cur_diffusion_t_idx']]
            for last_t_idx, lats_t2 in enumerate(last_ts):
                batches[last_traj_idx]['trans_t'] = torch.cat([
                    torch.cat([reorgnized_trans_1[last_traj_idx][i:i+1, :target_seq_len[i][last_traj_idx]], batches[last_traj_idx]['trans_t_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
                ], dim=0)
                batches[last_traj_idx]['rotmats_t'] = torch.cat([
                    torch.cat([reorgnized_rotmats_1[last_traj_idx][i:i+1, :target_seq_len[i][last_traj_idx]], batches[last_traj_idx]['rotmats_t_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
                ], dim=0)
                batches[last_traj_idx]['aatypes_t'] = torch.cat([
                    torch.cat([reorgnized_aatypes_1[last_traj_idx][i:i+1, :target_seq_len[i][last_traj_idx]], batches[cur_traj_idx]['pred_aatypes_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
                ], dim=0)

                t = torch.ones((num_batch, 1), device=self._device) * last_t1

                if self._cfg.provide_kappa:
                    batches[last_traj_idx]['binder_so3_t'] = self.rot_sample_kappa(t)
                else:
                    batches[last_traj_idx]['binder_so3_t'] = t
                batches[last_traj_idx]['binder_r3_t'] = t
                batches[last_traj_idx]['binder_cat_t'] = (1 - self._cfg.min_t) * torch.ones((num_batch, 1), device=self._device)
                batches[last_traj_idx]['target_so3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batches[last_traj_idx]['binder_so3_t'])
                batches[last_traj_idx]['target_r3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batches[last_traj_idx]['binder_r3_t'])
                batches[last_traj_idx]['target_cat_t'] = (1 - self._cfg.min_t) * torch.ones_like(batches[last_traj_idx]['binder_cat_t'])

                lats_dt = lats_t2 - last_t1

                with torch.no_grad():
                    model_out = model(batches[last_traj_idx])

                batches[last_traj_idx]['pred_trans_1'] = model_out['pred_trans']
                batches[last_traj_idx]['pred_rotmats_1'] = model_out['pred_rotmats']
                batches[last_traj_idx]['pred_aatypes_1'] = model_out['pred_aatypes']
                batches[last_traj_idx]['pred_logits_1'] = model_out['pred_logits']
                for i, num_res_i in enumerate(num_res):
                    clean_traj[max(0, t_idx - switch_steps * (len(batches) - 1)) + last_t_idx][i][0][str(last_traj_idx)] = frames_to_atom37(batches[last_traj_idx]['pred_trans_1'][i:i+1, -num_res_i:], batches[last_traj_idx]['pred_rotmats_1'][i:i+1, -num_res_i:])
                
                if self._cfg.self_condition:
                    batches[last_traj_idx]['trans_sc'] = _trans_diffuse_mask(
                        batches[last_traj_idx]['pred_trans_1'], reorgnized_trans_1[last_traj_idx], batches[last_traj_idx]['diffuse_mask'])
                    batches[last_traj_idx]['aatypes_sc'] = _trans_diffuse_mask(
                        batches[last_traj_idx]['pred_logits_1'], reorgnized_logits_1[last_traj_idx], batches[last_traj_idx]['diffuse_mask'])

                # Take reverse step            
                batches[last_traj_idx]['trans_t_2'] = self._trans_euler_step(
                    lats_dt, last_t1, batches[last_traj_idx]['pred_trans_1'], batches[last_traj_idx]['trans_t_1'])
                batches[last_traj_idx]['rotmats_t_2'] = self._rots_euler_step(
                    lats_dt, last_t1, batches[last_traj_idx]['pred_rotmats_1'], batches[last_traj_idx]['rotmats_t_1'])
                
                if self._aatypes_cfg.do_purity:
                    batches[last_traj_idx]['aatypes_t_2'] = self._aatypes_euler_step_purity(lats_dt, last_t1, batches[last_traj_idx]['pred_logits_1'], batches[last_traj_idx]['aatypes_t_1'])
                else:
                    batches[last_traj_idx]['aatypes_t_2'] = self._aatypes_euler_step(lats_dt, last_t1, batches[last_traj_idx]['pred_logits_1'], batches[last_traj_idx]['aatypes_t_1'])

                batches[last_traj_idx]['trans_t_2'] = _trans_diffuse_mask(batches[last_traj_idx]['trans_t_2'], reorgnized_trans_1[last_traj_idx], batches[last_traj_idx]['diffuse_mask'])
                batches[last_traj_idx]['rotmats_t_2'] = _rots_diffuse_mask(batches[last_traj_idx]['rotmats_t_2'], reorgnized_rotmats_1[last_traj_idx], batches[last_traj_idx]['diffuse_mask'])
                batches[last_traj_idx]['aatypes_t_2'] = _aatypes_diffuse_mask(batches[last_traj_idx]['aatypes_t_2'], reorgnized_aatypes_1[last_traj_idx], batches[last_traj_idx]['diffuse_mask'])
                batches[last_traj_idx]['trans_t_1'], batches[last_traj_idx]['rotmats_t_1'], batches[last_traj_idx]['aatypes_t_1'] = batches[last_traj_idx]['trans_t_2'], batches[last_traj_idx]['rotmats_t_2'], batches[last_traj_idx]['aatypes_t_2']
                for i, num_res_i in enumerate(num_res):
                    prot_traj[max(0, t_idx - switch_steps * (len(batches) - 1)) + last_t_idx + 1][i][0][str(last_traj_idx)] = frames_to_atom37(batches[last_traj_idx]['trans_t_2'][i:i+1, -num_res_i:], batches[last_traj_idx]['rotmats_t_2'][i:i+1, -num_res_i:])
                
                condition_t1 = condition_t2
                batches[next_traj_idx]['cur_diffusion_t_idx'] = max(0, t_idx - switch_steps * (len(batches) - 1)) + condition_t_idx + 1
        
            last_t_1 = ts[-1]

            batches[last_traj_idx]['trans_t'] = torch.cat([
                torch.cat([reorgnized_trans_1[last_traj_idx][i:i+1, :target_seq_len[i][last_traj_idx]], batches[last_traj_idx]['trans_t_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
            ], dim=0)
            batches[last_traj_idx]['rotmats_t'] = torch.cat([
                torch.cat([reorgnized_rotmats_1[last_traj_idx][i:i+1, :target_seq_len[i][last_traj_idx]], batches[last_traj_idx]['rotmats_t_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
            ], dim=0)
            batches[last_traj_idx]['aatypes_t'] = torch.cat([
                torch.cat([reorgnized_aatypes_1[last_traj_idx][i:i+1, :target_seq_len[i][last_traj_idx]], batches[cur_traj_idx]['pred_aatypes_1'][i:i+1, -num_res_i:]], dim=1) for i, num_res_i in enumerate(num_res)
            ], dim=0)

            with torch.no_grad():
                model_out = model(batches[last_traj_idx])
            batches[last_traj_idx]['pred_trans_1'] = model_out['pred_trans']
            batches[last_traj_idx]['pred_rotmats_1'] = model_out['pred_rotmats']
            batches[last_traj_idx]['pred_aatypes_1'] = model_out['pred_aatypes']
            
            for i, num_res_i in enumerate(num_res):
                clean_traj[-1][i][0][str(last_traj_idx)] = frames_to_atom37(batches[last_traj_idx]['pred_trans_1'][i:i+1, -num_res_i:], batches[last_traj_idx]['pred_rotmats_1'][i:i+1, -num_res_i:])
            for i, num_res_i in enumerate(num_res):
                prot_traj[-1][i][0][str(last_traj_idx)] = frames_to_atom37(batches[last_traj_idx]['pred_trans_1'][i:i+1, -num_res_i:], batches[last_traj_idx]['pred_rotmats_1'][i:i+1, -num_res_i:])

        return prot_traj, clean_traj
