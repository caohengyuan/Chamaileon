import torch
from torch import nn
from multiflow.models.utils import get_index_embedding, get_time_embedding


class TargetBinderNodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(TargetBinderNodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 2 + 1
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb
        if self._cfg.embed_aatype:
            self.aatype_embedding = nn.Embedding(21, self.c_s) # Always 21 because of 20 amino acids + 1 for unk
            embed_size += self.c_s + self._cfg.c_timestep_emb + self._cfg.aatype_pred_num_tokens
        if self._cfg.use_hotspot:
            embed_size += 1
        if self._cfg.use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(embed_size, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.LayerNorm(self.c_s),
            )
        else:
            self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(
            self,
            *,
            binder_so3_t,
            binder_r3_t,
            binder_cat_t,
            target_so3_t,
            target_r3_t,
            target_cat_t,
            res_mask,
            diffuse_mask,
            chain_index,
            pos,
            aatypes,
            aatypes_sc,
            target_seq_len,
            binder_seq_len,
            hotspot=None,
        ):
        # s: [b]

        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        target_binder_diffuse_mask = torch.ones_like(diffuse_mask)
        for i, (target_seq_len_i, binder_seq_len_i) in enumerate(zip(target_seq_len, binder_seq_len)):
            target_binder_diffuse_mask[i, :target_seq_len_i] = 0.0
            target_binder_diffuse_mask[i, target_seq_len_i:target_seq_len_i + binder_seq_len_i] = 1.0
        binder_so3_t_emb = self.embed_t(binder_so3_t, res_mask)
        binder_r3_t_emb = self.embed_t(binder_r3_t, res_mask)
        binder_cat_t_emb = self.embed_t(binder_cat_t, res_mask)
        target_so3_t_emb = self.embed_t(target_so3_t, res_mask)
        target_r3_t_emb = self.embed_t(target_r3_t, res_mask)
        target_cat_t_emb = self.embed_t(target_cat_t, res_mask)
        so3_t_emb = binder_so3_t_emb * target_binder_diffuse_mask[..., None] + target_so3_t_emb * (1 - target_binder_diffuse_mask[..., None])
        r3_t_emb = binder_r3_t_emb * target_binder_diffuse_mask[..., None] + target_r3_t_emb * (1 - target_binder_diffuse_mask[..., None])
        cat_t_emb = binder_cat_t_emb * target_binder_diffuse_mask[..., None] + target_cat_t_emb * (1 - target_binder_diffuse_mask[..., None])
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            so3_t_emb,
            r3_t_emb
        ]
        if self._cfg.embed_aatype:
            input_feats.append(self.aatype_embedding(aatypes))
            input_feats.append(cat_t_emb)
            input_feats.append(aatypes_sc)
        if self._cfg.embed_chain:
            input_feats.append(
                get_index_embedding(
                    chain_index,
                    self.c_pos_emb,
                    max_len=100
                )
            )
        if self._cfg.use_hotspot:
            if hotspot is None:
                raise ValueError("hotspot is required when use_hotspot = True.")
            input_feats.append(hotspot[..., None])
        return self.linear(torch.cat(input_feats, dim=-1))
