from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
import shutil
import re
import torch.distributed as dist
from collections import defaultdict
from glob import glob
from pytorch_lightning import LightningModule
from multiflow.analysis import utils as au
from multiflow.models.target_binder_flow_model import TargetBinderFlowModel
from multiflow.models import utils as mu
from multiflow.models import folding_model
from multiflow.data.target_binder_interpolant import TargetBinderInterpolant
from multiflow.data import utils as du
from multiflow.data import all_atom, so3_utils
from multiflow.data.residue_constants import restypes, restypes_with_x
from multiflow.data import residue_constants
from multiflow.experiments import utils as eu
from biotite.sequence.io import fasta
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
import gc


def split_hotspots_dict(hotspot_str):
    if not hotspot_str:
        return []

    items = hotspot_str.split(',')
    
    chain_groups = defaultdict(list)

    for item in items:
        match = re.match(r"([a-zA-Z]+)", item)
        if match:
            chain_id = match.group(1)
            chain_groups[chain_id].append(item)
    
    result = [",".join(parts) for chain_id, parts in chain_groups.items()]
    
    return result

class TargetBinderFlowModule(LightningModule):

    def __init__(self, cfg, dataset_cfg, folding_cfg=None, folding_device_id=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._dataset_cfg = dataset_cfg
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = TargetBinderFlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = TargetBinderInterpolant(cfg.interpolant)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

        self._folding_model = None
        self._folding_cfg = folding_cfg
        self._folding_device_id = folding_device_id

        self.aatype_pred_num_tokens = cfg.model.aatype_pred_num_tokens

    @property
    def folding_model(self):
        if self._folding_model is None:
            self._folding_model = folding_model.FoldingModel(
                self._folding_cfg,
                device_id=self._folding_device_id
            )
        return self._folding_model

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()


    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        if training_cfg.mask_plddt:
            loss_mask *= noisy_batch['plddt_mask']
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        gt_aatypes_1 = noisy_batch['aatypes_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 

        # Timestep used for normalization.
        binder_r3_t = noisy_batch['binder_r3_t'] # (B, 1)
        binder_so3_t = noisy_batch['binder_so3_t'] # (B, 1)
        binder_cat_t = noisy_batch['binder_cat_t'] # (B, 1)
        binder_r3_norm_scale = 1 - torch.min(
            binder_r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)
        binder_so3_norm_scale = 1 - torch.min(
            binder_so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)
        if training_cfg.aatypes_loss_use_likelihood_weighting:
            binder_cat_norm_scale = 1 - torch.min(
                binder_cat_t, torch.tensor(training_cfg.t_normalize_clip)) # (B, 1)
            assert binder_cat_norm_scale.shape == (num_batch, 1)
        else:
            binder_cat_norm_scale = 1.0

        target_r3_t = noisy_batch['target_r3_t'] # (B, 1)
        target_so3_t = noisy_batch['target_so3_t'] # (B, 1)
        target_cat_t = noisy_batch['target_cat_t'] # (B, 1)
        target_r3_norm_scale = 1 - torch.min(
            target_r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)
        target_so3_norm_scale = 1 - torch.min(
            target_so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)
        if training_cfg.aatypes_loss_use_likelihood_weighting:
            target_cat_norm_scale = 1 - torch.min(
                target_cat_t, torch.tensor(training_cfg.t_normalize_clip)) # (B, 1)
            assert target_cat_norm_scale.shape == (num_batch, 1)
        else:
            target_cat_norm_scale = 1.0

        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_logits = model_output['pred_logits'] # (B, N, aatype_pred_num_tokens)
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError('NaN encountered in pred_rots_vf')
        
        # aatypes loss
        target_binder_diffusion_mask = torch.ones_like(noisy_batch['diffuse_mask'])
        for i, (target_seq_len_i, binder_seq_len_i) in enumerate(zip(noisy_batch['target_seq_len'], noisy_batch['binder_seq_len'])):
            target_binder_diffusion_mask[i, :target_seq_len_i] = 0.0
            target_binder_diffusion_mask[i, target_seq_len_i:target_seq_len_i + binder_seq_len_i] = 1.0
        target_binder_diffusion_mask = target_binder_diffusion_mask[..., None] # (B, N, 1)
        ce_loss = torch.nn.functional.cross_entropy(
            pred_logits.reshape(-1, self.aatype_pred_num_tokens),
            gt_aatypes_1.flatten().long(),
            reduction='none',
        ).reshape(num_batch, num_res)
        ce_loss = ce_loss / binder_cat_norm_scale * target_binder_diffusion_mask[:, :, 0] + ce_loss / target_cat_norm_scale * (1 - target_binder_diffusion_mask[:, :, 0])
        aatypes_loss = torch.sum(ce_loss * loss_mask, dim=-1) / (loss_denom / 3)
        aatypes_loss *= training_cfg.aatypes_loss_weight

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms = gt_bb_atoms * training_cfg.bb_atom_scale / binder_r3_norm_scale[..., None] * target_binder_diffusion_mask[..., None] \
            + gt_bb_atoms * training_cfg.bb_atom_scale / target_r3_norm_scale[..., None] * (1 - target_binder_diffusion_mask)[..., None]
        pred_bb_atoms = pred_bb_atoms * training_cfg.bb_atom_scale / binder_r3_norm_scale[..., None] * target_binder_diffusion_mask[..., None] \
            + pred_bb_atoms * training_cfg.bb_atom_scale / target_r3_norm_scale[..., None] * (1 - target_binder_diffusion_mask)[..., None]
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / loss_denom

        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) * training_cfg.trans_scale
        trans_error = trans_error / binder_r3_norm_scale * target_binder_diffusion_mask + \
            trans_error / target_r3_norm_scale * (1 - target_binder_diffusion_mask)
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        trans_loss = torch.clamp(trans_loss, max=5)

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf)
        rots_vf_error = rots_vf_error / binder_so3_norm_scale * target_binder_diffusion_mask + \
            rots_vf_error / target_so3_norm_scale * (1 - target_binder_diffusion_mask)
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) + 1)

        se3_vf_loss = trans_loss + rots_vf_loss
        auxiliary_loss = (
            bb_atom_loss * training_cfg.aux_loss_use_bb_loss
            + dist_mat_loss * training_cfg.aux_loss_use_pair_loss
        )
        # TODO: here is not a good implementation?
        auxiliary_loss *= (
            (binder_r3_t[:, 0] > training_cfg.aux_loss_t_pass)
            & (binder_so3_t[:, 0] > training_cfg.aux_loss_t_pass)
            & (target_r3_t[:, 0] > training_cfg.aux_loss_t_pass)
            & (target_so3_t[:, 0] > training_cfg.aux_loss_t_pass)
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        auxiliary_loss = torch.clamp(auxiliary_loss, max=5)

        train_loss = trans_loss + rots_vf_loss + auxiliary_loss + aatypes_loss
        if torch.any(torch.isnan(train_loss)):
            raise ValueError('NaN loss encountered')
        self._prev_batch = noisy_batch
        self._prev_loss_denom = loss_denom
        self._prev_loss = {
            "trans_loss": trans_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "train_loss": train_loss,
            'aatypes_loss': aatypes_loss
        }
        return self._prev_loss

    def validation_step(self, batch: Any, batch_idx: int):
        gc.collect()
        torch.cuda.empty_cache()

        rank = self.global_rank

        res_mask = batch['res_mask']
        self.interpolant.set_device(res_mask.device)
        num_batch, _ = res_mask.shape
        num_res = batch['binder_seq_len']
        
        diffuse_mask = batch['diffuse_mask']
        json_idx = batch['json_idx']

        gt_aatypes = batch['aatypes_1'].clone()
        gt_trans_1 = batch['trans_1'].clone()
        gt_rotmats_1 = batch['rotmats_1'].clone()

        # assert (diffuse_mask == 1.0).all()

        model_dtype = next(self.model.parameters()).dtype
        model_device = next(self.model.parameters()).device
        for k, v in batch.items():
            if k in ['aatypes_1', 'res_mask', 'chain_idx', 'res_idx', 'plddt_mask', 'diffuse_mask', 'binder_seq_len', 'json_idx', 'target_seq_len', 'binder_seq_len', 'hotspot']:
                batch[k] = v.to(dtype=torch.int64, device=model_device)
            elif k in ['res_plddt', 'rotmats_1', 'trans_1']:
                batch[k] = v.to(dtype=model_dtype, device=model_device)
            elif k in ['name', 'target_hotspot_str', 'binder_hotspot_str']:
                pass
            else:
                raise ValueError(f'Unrecognized key {k} in batch')
        
        prot_traj, model_traj = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            trans_1=batch['trans_1'],
            rotmats_1=batch['rotmats_1'],
            aatypes_1=batch['aatypes_1'],
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            chain_idx=batch['chain_idx'],
            res_idx=batch['res_idx'],
            target_seq_len=batch['target_seq_len'],
            binder_seq_len=batch['binder_seq_len'],
            hotspot=batch.get('hotspot', None),
        )
        frames_to_atom37 = lambda x,y: all_atom.atom37_from_trans_rot(x, y, None).detach().cpu().numpy()

        targets = [frames_to_atom37(gt_trans_1[i:i+1, :target_seq_len_i], 
                                    gt_rotmats_1[i:i+1, :target_seq_len_i]) for i, target_seq_len_i in enumerate(batch['target_seq_len'])]
        gt_samples = [frames_to_atom37(gt_trans_1[i:i+1, -binder_seq_len_i:], 
                                    gt_rotmats_1[i:i+1, -binder_seq_len_i:]) for i, binder_seq_len_i in enumerate(batch['binder_seq_len'])]
        samples = [sample_i[0].numpy() for sample_i in prot_traj[-1]]
        generated_aatypes = [sample_i[1].numpy() for sample_i in prot_traj[-1]]

        batch_level_aatype_metrics = mu.calc_aatype_metrics_list_of_arrays(generated_aatypes)

        batch_metrics = []
        for i in range(num_batch):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f'sample_{json_idx[i].item()}_idx_{batch_idx}_len_{num_res[i]}_rank_{rank}_step_{self.trainer.global_step}'
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Write out sample to PDB file
            final_pos = samples[i][0]
            saved_path = au.write_prot_to_pdb(
                final_pos,
                os.path.join(sample_dir, 'sample.pdb'),
                no_indexing=True,
                aatype=generated_aatypes[i][0],
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            # Write out sample and target to a PDB file
            target_pos = targets[i][0]
            target_saved_path = au.write_prot_to_pdb(
                target_pos,
                os.path.join(sample_dir, 'target.pdb'),
                no_indexing=True,
                aatype=gt_aatypes[i][:batch['target_seq_len'][i]].detach().cpu().numpy(),
            )
            au.write_prot_to_pdb_multi_chain(
                prot_pos=np.concatenate([
                    target_pos,
                    final_pos,
                ], axis=0),
                file_path=saved_path.replace('.pdb', '_merged_with_target.pdb'),
                aatype=np.concatenate([
                    gt_aatypes[i][:batch['target_seq_len'][i]].detach().cpu().numpy(),
                    generated_aatypes[i][0],
                ], axis=0),
                no_indexing=True,
                chain_splits=[batch['target_seq_len'][i], num_res[i]],
                chain_ids=['A', 'B'],
            )

            # Write ground-truth structure
            gt_pos = gt_samples[i][0]
            gt_saved_path = au.write_prot_to_pdb(
                gt_pos,
                os.path.join(sample_dir, 'gt_sample.pdb'),
                no_indexing=True,
                aatype=gt_aatypes[i][-batch['binder_seq_len'][i]:].detach().cpu().numpy(),
            )
            au.write_prot_to_pdb_multi_chain(
                prot_pos=np.concatenate([
                    target_pos,
                    gt_samples[i][0],
                ], axis=0),
                file_path=gt_saved_path.replace('.pdb', '_merged_with_target.pdb'),
                aatype=np.concatenate([
                    gt_aatypes[i][:batch['target_seq_len'][i]].detach().cpu().numpy(),
                    gt_aatypes[i][-batch['binder_seq_len'][i]:].detach().cpu().numpy(),
                ], axis=0),
                no_indexing=True,
                chain_splits=[batch['target_seq_len'][i], num_res[i]],
                chain_ids=['A', 'B'],
            )

            # Run designability
            pmpnn_pdb_path = saved_path.replace('.pdb', '_pmpnn.pdb')
            shutil.copy(saved_path, pmpnn_pdb_path)
            pmpnn_fasta_path = self.run_pmpnn(
                sample_dir,
                pmpnn_pdb_path,
            )
            folded_dir = os.path.join(sample_dir, 'folded')
            os.makedirs(folded_dir, exist_ok=True)

            if self.interpolant._aatypes_cfg.corrupt:
                # Codesign
                codesign_fasta = fasta.FastaFile()
                codesign_fasta['codesign_seq_1'] = "".join([restypes_with_x[x] for x in generated_aatypes[i][0]])
                codesign_fasta_path = os.path.join(sample_dir, 'codesign.fa')
                codesign_fasta.write(codesign_fasta_path)
                gt_fasta = fasta.FastaFile()
                gt_fasta['gt_seq_1'] = "".join([restypes_with_x[x] for x in gt_aatypes[i][-num_res[i]:]])
                gt_fasta_path = os.path.join(sample_dir, 'ground_truth.fa')
                gt_fasta.write(gt_fasta_path)

                codesign_folded_output = self.folding_model.fold_fasta(codesign_fasta_path, folded_dir)
                codesign_results = mu.process_folded_outputs(saved_path, codesign_folded_output)

                # make a fasta file with a single PMPNN sequence to be folded
                reloaded_fasta = fasta.FastaFile.read(pmpnn_fasta_path)
                single_fasta = fasta.FastaFile()
                single_fasta['pmpnn_seq_1'] = reloaded_fasta['pmpnn_seq_1']
                single_fasta_path = os.path.join(sample_dir, 'pmpnn_single.fasta')
                single_fasta.write(single_fasta_path)

                single_pmpnn_folded_output = self.folding_model.fold_fasta(single_fasta_path, folded_dir)
                single_pmpnn_results = mu.process_folded_outputs(saved_path, single_pmpnn_folded_output)

                designable_metrics = {
                    'codesign_bb_rmsd': codesign_results.bb_rmsd.min(),
                    'pmpnn_bb_rmsd': single_pmpnn_results.bb_rmsd.min(),
                }
            else:
                raise ValueError('Should be using uncorrupted aatypes in MSDesign.')
                # Just structure
                folded_output = self.folding_model.fold_fasta(pmpnn_fasta_path, folded_dir)

                designable_results = mu.process_folded_outputs(saved_path, folded_output) 
                designable_metrics = {
                    'bb_rmsd': designable_results.bb_rmsd.min()
                }
            try:
                mdtraj_metrics = mu.calc_mdtraj_metrics(saved_path)
                ca_ca_metrics = mu.calc_ca_ca_metrics(final_pos[:, residue_constants.atom_order['CA']])
                batch_metrics.append((mdtraj_metrics | ca_ca_metrics | designable_metrics | batch_level_aatype_metrics))
            except Exception as e:
                print(e)
                continue

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)
        
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name,metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()

        gc.collect()
        torch.cuda.empty_cache()

        model_dtype = next(self.model.parameters()).dtype
        model_device = next(self.model.parameters()).device
        for k, v in batch.items():
            if k in ['aatypes_1', 'res_mask', 'chain_idx', 'res_idx', 'plddt_mask', 'diffuse_mask', 'binder_seq_len', 'json_idx', 'target_seq_len', 'binder_seq_len', 'hotspot']:
                batch[k] = v.to(dtype=torch.int64, device=model_device)
            elif k in ['res_plddt', 'rotmats_1', 'trans_1']:
                batch[k] = v.to(dtype=model_dtype, device=model_device)
            elif k in ['name', 'target_hotspot_str', 'binder_hotspot_str']:
                pass
            else:
                raise ValueError(f'Unrecognized key {k} in batch')

        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = (
                    model_sc['pred_trans'] * noisy_batch['diffuse_mask'][..., None]
                    + noisy_batch['trans_1'] * (1 - noisy_batch['diffuse_mask'][..., None])
                )
                logits_1 = torch.nn.functional.one_hot(
                    batch['aatypes_1'].long(), num_classes=self.aatype_pred_num_tokens).float()
                noisy_batch['aatypes_sc'] = (
                    model_sc['pred_logits'] * noisy_batch['diffuse_mask'][..., None]
                    + logits_1 * (1 - noisy_batch['diffuse_mask'][..., None])
                )
        batch_losses = self.model_step(noisy_batch)

        num_batch = batch_losses['train_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        binder_so3_t = torch.squeeze(noisy_batch['binder_so3_t'])
        self._log_scalar(
            "train/binder_so3_t",
            np.mean(du.to_numpy(binder_so3_t)),
            prog_bar=False, batch_size=num_batch)
        binder_r3_t = torch.squeeze(noisy_batch['binder_r3_t'])
        self._log_scalar(
            "train/binder_r3_t",
            np.mean(du.to_numpy(binder_r3_t)),
            prog_bar=False, batch_size=num_batch)
        binder_cat_t = torch.squeeze(noisy_batch['binder_cat_t'])
        self._log_scalar(
            "train/binder_cat_t",
            np.mean(du.to_numpy(binder_cat_t)),
            prog_bar=False, batch_size=num_batch)
        target_so3_t = torch.squeeze(noisy_batch['target_so3_t'])
        self._log_scalar(
            "train/target_so3_t",
            np.mean(du.to_numpy(target_so3_t)),
            prog_bar=False, batch_size=num_batch)
        target_r3_t = torch.squeeze(noisy_batch['target_r3_t'])
        self._log_scalar(
            "train/target_r3_t",
            np.mean(du.to_numpy(target_r3_t)),
            prog_bar=False, batch_size=num_batch)
        target_cat_t = torch.squeeze(noisy_batch['target_cat_t'])
        self._log_scalar(
            "train/target_cat_t",
            np.mean(du.to_numpy(target_cat_t)),
            prog_bar=False, batch_size=num_batch)
        if not self._model_cfg.noise_target:
            for loss_name, loss_dict in batch_losses.items():
                if loss_name == 'rots_vf_loss':
                    batch_t = binder_so3_t
                elif loss_name == 'train_loss':
                    continue
                elif loss_name == 'aatypes_loss':
                    batch_t = binder_cat_t
                else:
                    batch_t = binder_r3_t
                stratified_losses = mu.t_stratified_loss(
                    batch_t, loss_dict, loss_name=loss_name)
                for k,v in stratified_losses.items():
                    self._log_scalar(
                        f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        train_loss = total_losses['train_loss']
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )

    def predict_step(self, batch, batch_idx):
        start_time = time.time()

        del batch_idx  # Unused
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = TargetBinderInterpolant(self._infer_cfg.interpolant)
        interpolant.set_device(device)

        if 'json_idx' in batch:
            sample_ids = batch['json_idx'].squeeze().tolist()
        elif 'sample_id' in batch:
            sample_ids = batch['sample_id'].squeeze().tolist()
        else:
            sample_ids = [0]
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        num_batch = len(sample_ids)

        if self._infer_cfg.task == 'target_binder_eval':
            # Here we only implement the case of generating a binder given a target
            sample_length = batch['binder_seq_len']
            sample_dirs = [os.path.join(self.inference_dir, pdb_name, f'binder_seq_length_{sample_length_i}', f'json_idx_{json_idx_i.item()}')
                           for pdb_name, sample_length_i, json_idx_i in zip(batch['target_pdb_name'], sample_length, batch['json_idx'])]
            trans_1 = batch['trans_1']    # will be masked in interpolant.sample
            rotmats_1 = batch['rotmats_1']    # will be masked in interpolant.sample
            aatypes_1 = batch['aatypes_1']    # will be masked in interpolant.sample
            res_mask = batch['res_mask']    # will be masked in interpolant.sample
            if not self._model_cfg.noise_target:
                diffuse_mask = batch['diffuse_mask']
            else:
                diffuse_mask = torch.ones_like(batch['diffuse_mask'])
                for i, target_seq_len_i in enumerate(batch['target_seq_len']):
                    diffuse_mask[i, :target_seq_len_i] = 0.0
            # diffuse_mask = batch['diffuse_mask']
            true_aatypes = batch['aatypes_1'].clone().detach()
            true_bb_pos = all_atom.atom37_from_trans_rot(batch['trans_1'], batch['rotmats_1'])
            chain_idx = batch['chain_idx']
            res_idx = batch['res_idx']
            target_seq_len = batch['target_seq_len']
            binder_seq_len = batch['binder_seq_len']
            hotspot = batch.get('hotspot', None)
            target_hotspot_str = batch.get('target_hotspot_str', None)
            binder_hotspot_str = batch.get('binder_hotspot_str', None)
            target_hotspot_str_list = [split_hotspots_dict(target_hotspot_str_i) for target_hotspot_str_i in target_hotspot_str]
            target_seg_lens = batch['target_seg_lens']
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            # save ground-truth information
            for i, num_res_i in enumerate(sample_length):
                target_len = int(batch[f'target_seq_len'][i])
                target_start_idx = 0
                target_end_idx = int(target_start_idx + target_len)
                binder_start_idx = int(batch['target_seq_len'][i])
                binder_end_idx = int(binder_start_idx + num_res_i)

                os.makedirs(sample_dirs[i], exist_ok=True)
                # hotspot string
                with open(os.path.join(sample_dirs[i], 'target_hotspot.txt'), 'w', encoding='utf-8') as f:
                    if target_hotspot_str is not None:
                        f.write(target_hotspot_str[i])
                    else:
                        f.write('')
                for chain_id, target_hotspot_chain in enumerate(target_hotspot_str_list[i]):
                    with open(os.path.join(sample_dirs[i], f'target_hotspot_chain_{chain_id}.txt'), 'w', encoding='utf-8') as f:
                        f.write(target_hotspot_chain)
                with open(os.path.join(sample_dirs[i], 'binder_hotspot.txt'), 'w', encoding='utf-8') as f:
                    if binder_hotspot_str is not None:
                        f.write(binder_hotspot_str[i])
                    else:
                        f.write('')
                # ground-truth target sequence
                gt_target_fasta = fasta.FastaFile()
                gt_target_fasta[f'gt_target_seq_1'] = "".join([restypes_with_x[x] for x in true_aatypes[i][target_start_idx:target_end_idx]])
                gt_target_fasta_path = os.path.join(sample_dirs[i], f'ground_truth_target.fasta')
                gt_target_fasta.write(gt_target_fasta_path)
                segs_total_len = 0
                for chain_id, target_seg_len in enumerate(target_seg_lens[i].tolist()):
                    gt_target_fasta_chain = fasta.FastaFile()
                    gt_target_fasta_chain[f'gt_target_seq_1_chain_{chain_id}'] = "".join([restypes_with_x[x] for x in true_aatypes[i][segs_total_len:segs_total_len + target_seg_len]])
                    segs_total_len += target_seg_len
                    gt_target_fasta_chain_path = os.path.join(sample_dirs[i], f'ground_truth_target_chain_{chain_id}.fasta')
                    gt_target_fasta_chain.write(gt_target_fasta_chain_path)
                # ground-truth binder sequence
                gt_binder_fasta = fasta.FastaFile()
                gt_binder_fasta[f'gt_binder_seq_1'] = "".join([restypes_with_x[x] for x in true_aatypes[i][binder_start_idx:binder_end_idx]])
                gt_binder_fasta_path = os.path.join(sample_dirs[i], f'ground_truth_binder.fasta')
                gt_binder_fasta.write(gt_binder_fasta_path)
                # ground-truth target structure
                au.write_prot_to_pdb(
                    prot_pos=true_bb_pos[i][target_start_idx:target_end_idx].cpu().detach().numpy(),
                    file_path=os.path.join(sample_dirs[i], f'ground_truth_target.pdb'),
                    aatype=true_aatypes[i][target_start_idx:target_end_idx].cpu().detach().numpy(),
                    no_indexing=True,
                )
                segs_total_len = 0
                for chain_id, target_seg_len in enumerate(target_seg_lens[i].tolist()):
                    au.write_prot_to_pdb(
                        prot_pos=true_bb_pos[i][segs_total_len:segs_total_len + target_seg_len].cpu().detach().numpy(),
                        file_path=os.path.join(sample_dirs[i], f'ground_truth_target_chain_{chain_id}.pdb'),
                        aatype=true_aatypes[i][segs_total_len:segs_total_len + target_seg_len].cpu().detach().numpy(),
                        no_indexing=True,
                    )
                    segs_total_len += target_seg_len
                # ground-truth binder structure
                au.write_prot_to_pdb(
                    prot_pos=true_bb_pos[i][binder_start_idx:binder_end_idx].cpu().detach().numpy(),
                    file_path=os.path.join(sample_dirs[i], f'ground_truth_binder.pdb'),
                    aatype=true_aatypes[i][binder_start_idx:binder_end_idx].cpu().detach().numpy(),
                    no_indexing=True,
                )
                # ground-truth target and binder structure
                au.write_prot_to_pdb_multi_chain(
                    prot_pos=np.concatenate([
                        true_bb_pos[i][target_start_idx:target_end_idx].cpu().detach().numpy(),
                        true_bb_pos[i][binder_start_idx:binder_end_idx].cpu().detach().numpy()
                    ], axis=0),
                    file_path=os.path.join(sample_dirs[i], f'ground_truth_binder_merged_with_target.pdb'),
                    aatype=np.concatenate([
                        true_aatypes[i][:batch['target_seq_len'][i]].detach().cpu().numpy(),
                        true_aatypes[i][-batch['binder_seq_len'][i]:].detach().cpu().numpy(),
                    ], axis=0),
                    no_indexing=True,
                    chain_splits=[batch['target_seq_len'][i], batch['binder_seq_len'][i]],
                    chain_ids=['A', 'B'],
                )
                prot_pos_all_segs = []
                aatype_all_segs = []
                chain_splits_all_segs = []
                chain_ids_all_segs = []
                segs_total_len = 0
                for chain_id, target_seg_len in enumerate(target_seg_lens[i].tolist()):
                    prot_pos_all_segs.append(
                        true_bb_pos[i][segs_total_len:segs_total_len + target_seg_len].cpu().detach().numpy())
                    aatype_all_segs.append(
                        true_aatypes[i][segs_total_len:segs_total_len + target_seg_len].cpu().detach().numpy())
                    chain_splits_all_segs.append(target_seg_len)
                    chain_ids_all_segs.append(alphabet[chain_id])
                    segs_total_len += target_seg_len
                prot_pos_all_segs.append(true_bb_pos[i][binder_start_idx:binder_end_idx].cpu().detach().numpy())
                aatype_all_segs.append(true_aatypes[i][binder_start_idx:binder_end_idx].cpu().detach().numpy())
                chain_splits_all_segs.append(batch['binder_seq_len'][i])
                chain_ids_all_segs.append(alphabet[chain_id+1])
                au.write_prot_to_pdb_multi_chain(
                    prot_pos=np.concatenate(prot_pos_all_segs, axis=0),
                    file_path=os.path.join(sample_dirs[i], f'ground_truth_binder_merged_with_target_all_segs.pdb'),
                    aatype=np.concatenate(aatype_all_segs, axis=0),
                    no_indexing=True,
                    chain_splits=chain_splits_all_segs,
                    chain_ids=chain_ids_all_segs,
                )
        elif self._infer_cfg.task == 'multi_target_binder_eval':
            # multi-state design (multi-target and one binder)
            # i is for batch index; j is for target-binder pair index
            sample_length = [int(sum(binder_seq_len_i) / len(binder_seq_len_i)) for binder_seq_len_i in batch['binder_seq_len']]
            sample_dirs = [os.path.join(self.inference_dir, f'binder_seq_length_{sample_length_i}', f'json_idx_{json_idx_i.item()}')
                           for sample_length_i, json_idx_i in zip(sample_length, batch['json_idx'])]
            trans_1 = [[trans_1_j.unsqueeze(0) for trans_1_j in batch_trans_1_i] for batch_trans_1_i in batch['trans_1']]
            rotmats_1 = [[rotmats_1_j.unsqueeze(0) for rotmats_1_j in batch_rotmats_1_i] for batch_rotmats_1_i in batch['rotmats_1']]
            aatypes_1 = [[aatypes_1_j.unsqueeze(0) for aatypes_1_j in batch_aatypes_1_i] for batch_aatypes_1_i in batch['aatypes_1']]
            res_mask = [[res_mask_j.unsqueeze(0) for res_mask_j in batch_res_mask_i] for batch_res_mask_i in batch['res_mask']]
            if not self._model_cfg.noise_target:
                diffuse_mask = [[diffuse_mask_j.unsqueeze(0) for diffuse_mask_j in batch_diffuse_mask_i] for batch_diffuse_mask_i in batch['diffuse_mask']]
            else:
                diffuse_mask = [[torch.ones_like(diffuse_mask_j).unsqueeze(0) for diffuse_mask_j in batch_diffuse_mask_i] for batch_diffuse_mask_i in batch['diffuse_mask']]
                for i, batch_target_seq_len_i in enumerate(batch['target_seq_len']):
                    for j, target_seq_len_j in enumerate(batch_target_seq_len_i):
                        diffuse_mask[i][j][:target_seq_len_j] = 0.0
            # diffuse_mask = batch['diffuse_mask']
            true_aatypes = [[aatypes_1_j.clone().detach().unsqueeze(0) for aatypes_1_j in batch_aatypes_1_i] for batch_aatypes_1_i in batch['aatypes_1']]
            true_bb_pos = [[all_atom.atom37_from_trans_rot(trans_1_j.unsqueeze(0), rotmats_1_j.unsqueeze(0)) for trans_1_j, rotmats_1_j in zip(batch_trans_1_i, batch_rotmats_1_i)] for batch_trans_1_i, batch_rotmats_1_i in zip(batch['trans_1'], batch['rotmats_1'])]
            chain_idx = [[chain_idx_j.unsqueeze(0) for chain_idx_j in batch_chain_idx_i] for batch_chain_idx_i in batch['chain_idx']]
            res_idx = [[res_idx_j.unsqueeze(0) for res_idx_j in batch_res_idx_i] for batch_res_idx_i in batch['res_idx']]
            target_seq_len = [[target_seq_len_j for target_seq_len_j in batch_target_seq_len_i] for batch_target_seq_len_i in batch['target_seq_len']]
            binder_seq_len = [[binder_seq_len_j for binder_seq_len_j in batch_binder_seq_len_i] for batch_binder_seq_len_i in batch['binder_seq_len']]
            hotspot = [[hotspot_j.unsqueeze(0) for hotspot_j in batch_hotspot_i] for batch_hotspot_i in batch['hotspot']]    # binder hotspot is masked in dataset.__getitem__
            hotspot_str = batch['hotspot_str']
            target_hotspot_str = [[hotspot_str_j[0] for hotspot_str_j in batch_hotspot_str_i] for batch_hotspot_str_i in batch['hotspot_str']]
            binder_hotspot_str = [[hotspot_str_j[1] for hotspot_str_j in batch_hotspot_str_i] for batch_hotspot_str_i in batch['hotspot_str']]

            # save ground-truth information
            for i, (batch_target_seq_len_i, batch_binder_seq_len_i) in enumerate(zip(batch['target_seq_len'], batch['binder_seq_len'])):
                os.makedirs(sample_dirs[i], exist_ok=True)
                for j, (target_seq_len_j, binder_seq_len_j) in enumerate(zip(batch_target_seq_len_i, batch_binder_seq_len_i)):
                    target_start_idx = 0
                    target_end_idx = int(target_seq_len_j)
                    binder_start_idx = target_end_idx
                    binder_end_idx = int(binder_start_idx + binder_seq_len_j)

                    # hotspot string
                    with open(os.path.join(sample_dirs[i], f'target_hotspot_{j}.txt'), 'w', encoding='utf-8') as f:
                        if hotspot_str is not None:
                            f.write(hotspot_str[i][j][0])
                        else:
                            f.write('')
                    with open(os.path.join(sample_dirs[i], f'binder_hotspot_{j}.txt'), 'w', encoding='utf-8') as f:
                        if hotspot_str is not None:
                            f.write(hotspot_str[i][j][1])
                        else:
                            f.write('')
                    # ground-truth target sequence
                    gt_target_fasta = fasta.FastaFile()
                    gt_target_fasta[f'gt_target_seq_{j}'] = "".join([restypes_with_x[x] for x in true_aatypes[i][j][0][target_start_idx:target_end_idx]])
                    gt_target_fasta_path = os.path.join(sample_dirs[i], f'ground_truth_target_{j}.fasta')
                    gt_target_fasta.write(gt_target_fasta_path)
                    # ground-truth binder sequence
                    gt_binder_fasta = fasta.FastaFile()
                    gt_binder_fasta[f'gt_binder_seq_{j}'] = "".join([restypes_with_x[x] for x in true_aatypes[i][j][0][binder_start_idx:binder_end_idx]])
                    gt_binder_fasta_path = os.path.join(sample_dirs[i], f'ground_truth_binder_{j}.fasta')
                    gt_binder_fasta.write(gt_binder_fasta_path)
                    # ground-truth target structure
                    au.write_prot_to_pdb(
                        prot_pos=true_bb_pos[i][j][0][target_start_idx:target_end_idx].cpu().detach().numpy(),
                        file_path=os.path.join(sample_dirs[i], f'ground_truth_target_{j}.pdb'),
                        aatype=true_aatypes[i][j][0][target_start_idx:target_end_idx].cpu().detach().numpy(),
                        no_indexing=True,
                    )
                    # ground-truth binder structure
                    au.write_prot_to_pdb(
                        prot_pos=true_bb_pos[i][j][0][binder_start_idx:binder_end_idx].cpu().detach().numpy(),
                        file_path=os.path.join(sample_dirs[i], f'ground_truth_binder_{j}.pdb'),
                        aatype=true_aatypes[i][j][0][binder_start_idx:binder_end_idx].cpu().detach().numpy(),
                        no_indexing=True,
                    )
                    # ground-truth target and binder structure
                    au.write_prot_to_pdb_multi_chain(
                        prot_pos=np.concatenate([
                            true_bb_pos[i][j][0][target_start_idx:target_end_idx].cpu().detach().numpy(),
                            true_bb_pos[i][j][0][binder_start_idx:binder_end_idx].cpu().detach().numpy()
                        ], axis=0),
                        file_path=os.path.join(sample_dirs[i], f'ground_truth_binder_{j}_merged_with_target_{j}.pdb'),
                        aatype=np.concatenate([
                            true_aatypes[i][j][0][:target_seq_len_j].detach().cpu().numpy(),
                            true_aatypes[i][j][0][-binder_seq_len_j:].detach().cpu().numpy(),
                        ], axis=0),
                        no_indexing=True,
                        chain_splits=[target_seq_len_j, binder_seq_len_j],
                        chain_ids=['A', 'B'],
                    )
        else:
            raise ValueError(f'Unknown task {self._infer_cfg.task}')

        # Skip runs if already exist
        top_sample_csv_paths = [os.path.join(sample_dir, 'top_sample.csv')
                                for sample_dir in sample_dirs]
        if all([os.path.exists(top_sample_csv_path) for top_sample_csv_path in top_sample_csv_paths]):
            self._print_logger.info(f'Skipping instance {sample_ids} length {sample_length}')
            return
        # Sample batch
        if self._infer_cfg.task == 'target_binder_eval':
            prot_traj, model_traj = interpolant.sample(
                num_batch, sample_length, self.model,
                trans_1=trans_1, rotmats_1=rotmats_1, aatypes_1=aatypes_1,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                chain_idx=chain_idx,
                res_idx=res_idx,
                target_seq_len=target_seq_len,
                binder_seq_len=binder_seq_len,
                hotspot=hotspot,
                sample_dir=os.path.join(sample_dirs[0], 'beam_search'), # assume batch size = 1
                target_seg_lens=target_seg_lens,
                hotspot_str=target_hotspot_str,
            )
            diffuse_mask = diffuse_mask if diffuse_mask is not None else torch.ones(1, sample_length)

            samples = [sample_i[0].numpy() for sample_i in prot_traj[-1]]
            generated_aatypes = [sample_i[1].numpy() for sample_i in prot_traj[-1]]

            for i in range(len(samples)):
                target_len = int(batch[f'target_seq_len'][i])
                target_start_idx = 0
                target_end_idx = int(target_start_idx + target_len)
                binder_start_idx = int(batch['target_seq_len'][i])
                binder_end_idx = int(binder_start_idx + sample_length[i])

                # generated binder sequence
                pred_fasta = fasta.FastaFile()
                pred_fasta['pred_seq_1'] = "".join([restypes_with_x[x] for x in generated_aatypes[i][0]])
                pred_fasta_path = os.path.join(sample_dirs[i], f'generated_binder.fasta')
                pred_fasta.write(pred_fasta_path)
                # generated binder structure
                final_pos = samples[i][0]
                generated_binder_pdb_path = au.write_prot_to_pdb(
                    final_pos,
                    os.path.join(sample_dirs[i], f'generated_binder.pdb'),
                    aatype=generated_aatypes[i][0],
                    no_indexing=True,
                )
                au.write_prot_to_pdb_multi_chain(
                    prot_pos=np.concatenate([
                        true_bb_pos[i][target_start_idx:target_end_idx].cpu().detach().numpy(),
                        final_pos,
                    ], axis=0),
                    file_path=generated_binder_pdb_path.replace('.pdb', '_merged_with_target.pdb'),
                    aatype=np.concatenate([
                        true_aatypes[i][target_start_idx:target_end_idx].cpu().detach().numpy(),
                        generated_aatypes[i][0],
                    ], axis=0),
                    no_indexing=True,
                    chain_splits=[target_end_idx - target_start_idx, final_pos.shape[0]],
                    chain_ids=['A', 'B'],
                )

                generated_prot_pos_all_segs = []
                generated_aatype_all_segs = []
                generated_chain_splits_all_segs = []
                generated_chain_ids_all_segs = []
                segs_total_len = 0
                for chain_id, target_seg_len in enumerate(target_seg_lens[i].tolist()):
                    generated_prot_pos_all_segs.append(
                        true_bb_pos[i][segs_total_len:segs_total_len + target_seg_len].cpu().detach().numpy())
                    generated_aatype_all_segs.append(
                        true_aatypes[i][segs_total_len:segs_total_len + target_seg_len].cpu().detach().numpy())
                    generated_chain_splits_all_segs.append(target_seg_len)
                    generated_chain_ids_all_segs.append(alphabet[chain_id])
                    segs_total_len += target_seg_len
                generated_prot_pos_all_segs.append(final_pos)
                generated_aatype_all_segs.append(generated_aatypes[i][0])
                generated_chain_splits_all_segs.append(batch['binder_seq_len'][i])
                generated_chain_ids_all_segs.append(alphabet[chain_id+1])
                au.write_prot_to_pdb_multi_chain(
                    prot_pos=np.concatenate(generated_prot_pos_all_segs, axis=0),
                    file_path=generated_binder_pdb_path.replace('.pdb', '_merged_with_target_all_segs.pdb'),
                    aatype=np.concatenate(generated_aatype_all_segs, axis=0),
                    no_indexing=True,
                    chain_splits=generated_chain_splits_all_segs,
                    chain_ids=generated_chain_ids_all_segs,
                )

                # use ProteinMPNN to redesign sequence
                inverse_folding_dir = os.path.join(sample_dirs[i], 'inverse_folding')
                os.makedirs(inverse_folding_dir, exist_ok=True)
                generated_binder_merged_with_target_all_segs_pdb_path = generated_binder_pdb_path.replace('.pdb', '_merged_with_target_all_segs.pdb')
                generated_binder_pmpnn_pdb_path = os.path.join(inverse_folding_dir, os.path.basename(generated_binder_merged_with_target_all_segs_pdb_path).replace('.pdb', '_pmpnn.pdb'))
                shutil.copy(generated_binder_merged_with_target_all_segs_pdb_path, generated_binder_pmpnn_pdb_path)
                generated_binder_pmpnn_fasta_path = self.run_pmpnn_modified_by_chy(
                    inverse_folding_dir,
                    generated_binder_pmpnn_pdb_path.replace('.pdb', '.jsonl'),
                    binder_seq_len[i],
                )

        elif self._infer_cfg.task == 'multi_target_binder_eval':
            prot_traj, model_traj = interpolant.multi_target_sample(
                num_batch, sample_length, self.model,
                trans_1=trans_1, rotmats_1=rotmats_1, aatypes_1=aatypes_1,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                chain_idx=chain_idx,
                res_idx=res_idx,
                target_seq_len=target_seq_len,
                binder_seq_len=binder_seq_len,
                hotspot=hotspot,
                sample_dir=os.path.join(sample_dirs[0], 'beam_search'), # assume batch size = 1
                hotspot_str=target_hotspot_str,
            )

            samples = [sample_i[0] for sample_i in prot_traj[-1]]
            generated_aatypes = [sample_i[1].numpy().astype(np.int64) for sample_i in prot_traj[-1]]

            for i in range(len(samples)):
                # generated binder sequence
                pred_fasta = fasta.FastaFile()
                pred_fasta['pred_seq_1'] = "".join([restypes_with_x[int(x)] for x in generated_aatypes[i][0]])
                pred_fasta_path = os.path.join(sample_dirs[i], f'generated_binder.fasta')
                pred_fasta.write(pred_fasta_path)
                
                for j, target_seq_len_j in enumerate(batch['target_seq_len'][i]):
                    target_len = int(batch[f'target_seq_len'][i][j])
                    target_start_idx = 0
                    target_end_idx = int(target_start_idx + target_len)
                    binder_start_idx = int(batch['target_seq_len'][i][j])
                    binder_end_idx = int(binder_start_idx + sample_length[i])

                    # generated binder structure
                    final_pos = samples[i][str(j)].numpy()[0]
                    generated_binder_pdb_path = au.write_prot_to_pdb(
                        final_pos,
                        os.path.join(sample_dirs[i], f'generated_binder_{j}.pdb'),
                        aatype=generated_aatypes[i][0],
                        no_indexing=True,
                    )

                    # merge binder with target
                    au.write_prot_to_pdb_multi_chain(
                        prot_pos=np.concatenate([
                            true_bb_pos[i][j][0, target_start_idx:target_end_idx].cpu().detach().numpy(),
                            final_pos,
                        ], axis=0),
                        file_path=generated_binder_pdb_path.replace('.pdb', f'_merged_with_target_{j}.pdb'),
                        aatype=np.concatenate([
                            true_aatypes[i][j][0, target_start_idx:target_end_idx].cpu().detach().numpy(),
                            generated_aatypes[i][0],
                        ], axis=0),
                        no_indexing=True,
                        chain_splits=[target_end_idx - target_start_idx, final_pos.shape[0]],
                        chain_ids=['A', 'B'],
                    )

        else:
            raise ValueError(f'Unknown task {self._infer_cfg.task}')
        
        end_time = time.time()
        inference_time = end_time - start_time

    def run_pmpnn(
            self,
            write_dir,
            pdb_input_path,
        ):
        self.folding_model.run_pmpnn(
            write_dir,
            pdb_input_path,
        )
        mpnn_fasta_path = os.path.join(
            write_dir,
            'seqs',
            os.path.basename(pdb_input_path).replace('.pdb', '.fa')
        )
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        all_header_seqs = [
            (f'pmpnn_seq_{i}', seq) for i, (_, seq) in enumerate(fasta_seqs.items())
            if i > 0
        ]
        modified_fasta_path = mpnn_fasta_path.replace('.fa', '_modified.fasta')
        fasta.FastaFile.write_iter(modified_fasta_path, all_header_seqs)
        return modified_fasta_path
    
    def run_pmpnn_modified_by_chy(
            self,
            pdb_dir,
            output_jsonl_path,
            binder_seq_len,
        ):
        self.folding_model.run_pmpnn(
            pdb_dir,
            output_jsonl_path,
        )
        mpnn_fasta_path = os.path.join(
            pdb_dir,
            'seqs',
            os.path.basename(output_jsonl_path).replace('.jsonl', '.fa')
        )
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        all_header_seqs = [
            (f'pmpnn_seq_{i}', seq[-binder_seq_len:]) for i, (_, seq) in enumerate(fasta_seqs.items())
            if i > 0
        ]
        modified_fasta_path = mpnn_fasta_path.replace('.fa', '_modified.fasta')
        fasta.FastaFile.write_iter(modified_fasta_path, all_header_seqs)
        return modified_fasta_path


    def compute_sample_metrics(self, batch, model_traj, bb_traj, aa_traj,
                               clean_aa_traj, true_bb_pos, true_aa, diffuse_mask,
                               sample_id, sample_length, sample_dir,
                               aatypes_corrupt,
                               also_fold_pmpnn_seq, write_sample_trajectories):

        noisy_traj_length, sample_length, _, _ = bb_traj.shape
        clean_traj_length = model_traj.shape[0]
        assert bb_traj.shape == (noisy_traj_length, sample_length, 37, 3)
        assert model_traj.shape == (clean_traj_length, sample_length, 37, 3)
        assert aa_traj.shape == (noisy_traj_length, sample_length)
        assert clean_aa_traj.shape == (clean_traj_length, sample_length)


        os.makedirs(sample_dir, exist_ok=True)

        traj_paths = eu.save_traj(
            bb_traj[-1],
            bb_traj,
            np.flip(model_traj, axis=0),
            du.to_numpy(diffuse_mask)[0],
            output_dir=sample_dir,
            aa_traj=aa_traj, 
            clean_aa_traj = clean_aa_traj,
            write_trajectories=write_sample_trajectories,
        )

        # Run PMPNN to get sequences
        sc_output_dir = os.path.join(sample_dir, 'self_consistency')
        os.makedirs(sc_output_dir, exist_ok=True)
        pdb_path = traj_paths['sample_path']
        pmpnn_pdb_path = os.path.join(
            sc_output_dir, os.path.basename(pdb_path))
        shutil.copy(pdb_path, pmpnn_pdb_path)
        assert (diffuse_mask == 1.0).all()
        pmpnn_fasta_path = self.run_pmpnn(
            sc_output_dir,
            pmpnn_pdb_path,
        )

        os.makedirs(os.path.join(sc_output_dir, 'codesign_seqs'), exist_ok=True)
        codesign_fasta = fasta.FastaFile()
        codesign_fasta['codesign_seq_1'] = "".join([restypes[x] for x in aa_traj[-1]])
        codesign_fasta_path = os.path.join(sc_output_dir, 'codesign_seqs', 'codesign.fa')
        codesign_fasta.write(codesign_fasta_path)


        folded_dir = os.path.join(sc_output_dir, 'folded')
        if os.path.exists(folded_dir):
            shutil.rmtree(folded_dir)
        os.makedirs(folded_dir, exist_ok=False)
        if aatypes_corrupt:
            # codesign metrics
            folded_output = self.folding_model.fold_fasta(codesign_fasta_path, folded_dir)

            if also_fold_pmpnn_seq:
                pmpnn_folded_output = self.folding_model.fold_fasta(pmpnn_fasta_path, folded_dir)
                pmpnn_results = mu.process_folded_outputs(pdb_path, pmpnn_folded_output, true_bb_pos)
                pmpnn_results.to_csv(os.path.join(sample_dir, 'pmpnn_results.csv'))

        else:
            # non-codesign metrics
            folded_output = self.folding_model.fold_fasta(pmpnn_fasta_path, folded_dir)

        mpnn_results = mu.process_folded_outputs(pdb_path, folded_output, true_bb_pos)


        if true_aa is not None:
            assert true_aa.shape == (1, sample_length)

            true_aa_fasta = fasta.FastaFile()
            true_aa_fasta['seq_1'] = "".join([restypes_with_x[i] for i in true_aa[0]])
            true_aa_fasta.write(os.path.join(sample_dir, 'true_aa.fa'))

            seq_recovery = (torch.from_numpy(aa_traj[-1]).to(true_aa[0].device) == true_aa[0]).float().mean()
            mpnn_results['inv_fold_seq_recovery'] = seq_recovery.item()

            # get seq recovery for PMPNN as well
            pmpnn_fasta = fasta.FastaFile.read(pmpnn_fasta_path)
            pmpnn_fasta_str = pmpnn_fasta['pmpnn_seq_1']
            pmpnn_fasta_idx = torch.tensor([restypes_with_x.index(x) for x in pmpnn_fasta_str]).to(true_aa[0].device)
            pmpnn_seq_recovery = (pmpnn_fasta_idx == true_aa[0]).float().mean()
            pmpnn_results['pmpnn_seq_recovery'] = pmpnn_seq_recovery.item()
            pmpnn_results.to_csv(os.path.join(sample_dir, 'pmpnn_results.csv'))
            mpnn_results['pmpnn_seq_recovery'] = pmpnn_seq_recovery.item()
            mpnn_results['pmpnn_bb_rmsd'] = pmpnn_results['bb_rmsd']

        # Save results to CSV
        mpnn_results.to_csv(os.path.join(sample_dir, 'sc_results.csv'))
        mpnn_results['length'] = sample_length
        mpnn_results['sample_id'] = sample_id
        del mpnn_results['header']
        del mpnn_results['sequence']

        # Select the top sample
        top_sample = mpnn_results.sort_values('bb_rmsd', ascending=True).iloc[:1]

        # Compute secondary structure metrics
        sample_dict = top_sample.iloc[0].to_dict()
        ss_metrics = mu.calc_mdtraj_metrics(sample_dict['sample_path'])
        top_sample['helix_percent'] = ss_metrics['helix_percent']
        top_sample['strand_percent'] = ss_metrics['strand_percent']
        return top_sample
