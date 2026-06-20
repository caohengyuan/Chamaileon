import logging
import json
import tree
import torch
import random
import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from multiflow.data import utils as du


def mask_to_ranges(mask, seg_lens=None, one_based=True):
    """
    将布尔掩码转换为带有链信息的范围列表。
    
    Args:
        mask: 布尔数组，True 表示选中的残基。
        seg_lens: 列表，表示每条链的长度 (例如 [100, 50] 表示 A链100长, B链50长)。
                  如果为 None，默认视为单条链。
    
    Returns:
        ranges: 列表，每个元素为 (chain_id, start, end)。
                chain_id 从 'A' 开始。start 和 end 是基于 1-based 的链内索引。
    """
    if seg_lens is None:
        seg_lens = [len(mask)]

    ranges = []
    
    # 验证 mask 长度是否匹配
    if sum(seg_lens) != len(mask):
        raise ValueError(f"Mask length ({len(mask)}) does not match sum of seg_lens ({sum(seg_lens)})")

    global_idx = 0
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for chain_idx, length in enumerate(seg_lens):
        chain_id = alphabet[chain_idx] if chain_idx < len(alphabet) else f"Chain{chain_idx}"
        
        # 获取当前链的 mask 片段
        chain_mask = mask[global_idx : global_idx + length]
        global_idx += length

        # 在当前链内寻找连续区域
        in_range = False
        start_local = -1
        
        for i, is_selected in enumerate(chain_mask):
            if one_based:
                # 转换为 1-based 索引
                current_residue_num = i + 1
            else:
                # 0-based 索引
                current_residue_num = i
            
            if is_selected:
                if not in_range:
                    in_range = True
                    start_local = current_residue_num
            else:
                if in_range:
                    in_range = False
                    # 记录范围: (Chain, Start, End)
                    # End 是上一个残基，即 current_residue_num - 1
                    ranges.append((chain_id, start_local, current_residue_num - 1))
        
        # 处理链末尾的情况
        if in_range:
            ranges.append((chain_id, start_local, length))

    return ranges

def ranges_to_str(ranges):
    """
    将带有链信息的范围列表转换为字符串。
    
    Args:
        ranges: 列表，格式为 [(chain_id, start, end), ...]
    
    Returns:
        string: 格式如 "A1-10,A15,B5-8"
    """
    s = []
    for chain_id, start, end in ranges:
        if start == end:
            s.append(f"{chain_id}{start}")
        else:
            s.append(f"{chain_id}{start}-{end}")
    return ",".join(s)


class PDBTargetBinderEvalDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
        ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

        # Process clusters
        self.raw_json = json.load(open(self._dataset_cfg.json_path, 'r'))
        print('NOTICE: skip filtering for demo purposes')
        metadata_json = self.raw_json
        metadata_json = metadata_json * self._dataset_cfg.num_samples
        # metadata_json = self._filter_metadata(self.raw_json)
        metadata_json = sorted(metadata_json, key=lambda x: x['target']['modeled_seq_len'] + x['binder']['modeled_seq_len'], reverse=True)

        self._pdb_to_cluster = self._read_clusters(self._dataset_cfg.cluster_path, synthetic=False)
        self._max_cluster = max(self._pdb_to_cluster.values())
        self._missing_pdbs = 0
        def cluster_lookup(pdb):
            pdb = pdb.upper()
            if pdb not in self._pdb_to_cluster:
                self._pdb_to_cluster[pdb] = self._max_cluster + 1
                self._max_cluster += 1
                self._missing_pdbs += 1
            return self._pdb_to_cluster[pdb]
        for item in metadata_json:
            for chain in ['target', 'binder']:
                item[chain].update({'cluster': cluster_lookup(item[chain]['pdb_name'])})
        
        # remove redesigned and synthetic data

        self._create_split(metadata_json)

        # remove dataset_cfg.test_set_pdb_ids_path is not None

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg
    
    def __len__(self):
        return len(self.json)

    def _read_clusters(self, cluster_path, synthetic=False):
        pdb_to_cluster = {}
        with open(cluster_path, "r") as f:
            for i,line in enumerate(f):
                for chain in line.split(' '):
                    if not synthetic:
                        pdb = chain.split('_')[0].strip()
                    else:
                        pdb = chain.strip()
                    pdb_to_cluster[pdb.upper()] = i
        return pdb_to_cluster
    
    def _rog_filter_fn(self, df, quantile):
        y_quant = pd.pivot_table(
            df,
            values='radius_gyration', 
            index='modeled_seq_len',
            aggfunc=lambda x: np.quantile(x, quantile)
        )
        x_quant = y_quant.index.to_numpy()
        y_quant = y_quant.radius_gyration.to_numpy()

        # Fit polynomial regressor
        poly = PolynomialFeatures(degree=4, include_bias=True)
        poly_features = poly.fit_transform(x_quant[:, None])
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(poly_features, y_quant)

        # Calculate cutoff for all sequence lengths
        max_len = df.modeled_seq_len.max()
        pred_poly_features = poly.fit_transform(np.arange(max_len)[:, None])
        # Add a little more.
        pred_y = poly_reg_model.predict(pred_poly_features) + 0.1

        # save cutoff dict
        cutoff_dict = {seq_len: pred_y[seq_len - 1] for seq_len in range(1, max_len + 1)}

        # return filtering function
        def is_valid(seq_len, rog):
            if seq_len not in cutoff_dict:
                raise ValueError(f"Sequence length {seq_len} not in cutoff dict.")
            return rog < cutoff_dict[seq_len]

        return is_valid

    def _filter_metadata(self, metadata_json):
        """Filter metadata."""
        filter_cfg = self._dataset_cfg.filter
        filtered_metadata_json = []

        if filter_cfg.rog_quantile is not None:
            rog_quantile_fn = {}
            for chain in filter_cfg.which_chains:
                df = pd.DataFrame([x[chain] for x in metadata_json])
                rog_quantile_fn[chain] = self._rog_filter_fn(df, filter_cfg.rog_quantile)

        for item in metadata_json:
            flag = True
            for chain in filter_cfg.which_chains:
                if filter_cfg.oligomeric is not None and item[chain]['oligomeric_detail'] not in filter_cfg.oligomeric:
                    flag = False
                    break
                if filter_cfg.num_chains is not None and item[chain]['num_chains'] not in filter_cfg.num_chains:
                    flag = False
                    break
                if filter_cfg.min_num_res is not None and filter_cfg.max_num_res is not None and not (filter_cfg.min_num_res <= item[chain]['modeled_seq_len'] <= filter_cfg.max_num_res):
                    flag = False
                    break
                if filter_cfg.max_coil_percent is not None and not (item[chain]['coil_percent'] <= filter_cfg.max_coil_percent):
                    flag = False
                    break
                if filter_cfg.rog_quantile is not None and not rog_quantile_fn[chain](item[chain]['modeled_seq_len'], item[chain]['radius_gyration']):
                    flag = False
                    break
            if flag:
                filtered_metadata_json.append(item)

        return filtered_metadata_json
    
    def _create_split(self, data_json):
        # Training or validation specific logic.
        if self.is_training:
            self.json = data_json
            self._log.info(
                f'Training: {len(self.json)} examples')
        else:
            if self._dataset_cfg.max_eval_length is None:
                eval_lengths = np.array([item['binder']['modeled_seq_len'] for item in data_json])
            else:
                eval_lengths = np.array([item['binder']['modeled_seq_len'] for item in data_json if item['binder']['modeled_seq_len'] <= self._dataset_cfg.max_eval_length])
            all_lengths = np.sort(np.unique(eval_lengths))
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.dataset_cfg.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_json = [item for item in data_json if item['binder']['modeled_seq_len'] in eval_lengths]

            # Group by 'binder.modeled_seq_len' in list of dicts and sample

            # Build mapping: seq_len -> list of items
            seq_len_to_items = defaultdict(list)
            for item in eval_json:
                seq_len = item['binder']['modeled_seq_len']
                seq_len_to_items[seq_len].append(item)

            # For each eval length, sample items
            sampled_items = []
            rng = np.random.default_rng(123)
            for seq_len in eval_lengths:
                items = seq_len_to_items.get(seq_len, [])
                if len(items) == 0:
                    continue
                n = self._dataset_cfg.samples_per_eval_length
                if len(items) < n:
                    # Sample with replacement if not enough items
                    idxs = rng.choice(len(items), n, replace=True)
                else:
                    idxs = rng.choice(len(items), n, replace=False)
                for idx in idxs:
                    sampled_items.append(items[idx])

            # Sort by seq_len descending
            sampled_items.sort(key=lambda x: x['binder']['modeled_seq_len'], reverse=True)
            self.json = sampled_items
            self._log.info(
                f'Validation: {len(self.json)} examples with lengths {eval_lengths}')
        for idx, item in enumerate(self.json):
            item.update({'index': idx})
            if not self.is_training:
                print(f"VALIDATION item {idx}: {item['binder']['processed_path']}")

    def _process_json_item(self, processed_file_path):
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_chain_feats(processed_feats, scale_factor=1.0, center=False)

        # Only take modeled residues.
        modeled_idx = processed_feats['modeled_idx']
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        del processed_feats['modeled_idx']
        processed_feats = tree.map_structure(
            lambda x: x[min_idx:(max_idx+1)], processed_feats)

        # Run through OpenFold data transforms.
        chain_feats = {
            'aatype': torch.tensor(processed_feats['aatype']).long(),
            'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
            'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()
        res_plddt = processed_feats['b_factors'][:, 1]
        res_mask = torch.tensor(processed_feats['bb_mask']).int()

        # Re-number residue indices for each chain such that it starts from 1.
        # Randomize chain indices.
        chain_idx = processed_feats['chain_index']
        res_idx = processed_feats['residue_index']
        segment_idx = processed_feats['segment_index']
        new_res_idx = np.zeros_like(res_idx)
        new_chain_idx = np.zeros_like(res_idx)
        all_chain_idx = np.unique(chain_idx).tolist()
        shuffled_chain_idx = np.array(
            random.sample(all_chain_idx, len(all_chain_idx))) - np.min(all_chain_idx) + 1
        for i,chain_id in enumerate(all_chain_idx):
            chain_mask = (chain_idx == chain_id).astype(int)
            chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
            new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

            # Shuffle chain_index
            replacement_chain_id = shuffled_chain_idx[i]
            new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

        if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
            raise ValueError(f'Found NaNs in {processed_file_path}')

        if self._dataset_cfg.use_hotspot:
            return {
                'res_plddt': res_plddt,
                'aatypes_1': chain_feats['aatype'],
                'rotmats_1': rotmats_1,
                'trans_1': trans_1,
                'res_mask': res_mask,
                'chain_idx': new_chain_idx,
                'segment_idx': segment_idx,
                'res_idx': new_res_idx,
                'hotspot': torch.tensor(processed_feats['hotspot_mask']).int()
            }
        else:
            return {
                'res_plddt': res_plddt,
                'aatypes_1': chain_feats['aatype'],
                'rotmats_1': rotmats_1,
                'trans_1': trans_1,
                'res_mask': res_mask,
                'chain_idx': new_chain_idx,
                'segment_idx': segment_idx,
                'res_idx': new_res_idx,
            }

    def process_json_item(self, data):
        path = os.path.join(data['processed_path'])
        seq_len = data['modeled_seq_len']
        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]
        processed_item = self._process_json_item(path)
        processed_item['pdb_name'] = data['pdb_name']
        processed_item['seq_len'] = seq_len
        aatypes_1 = du.to_numpy(processed_item['aatypes_1'])
        if use_cache:
            self._cache[path] = processed_item
        return processed_item

    def _add_plddt_mask(self, feats, plddt_threshold):
        feats['plddt_mask'] = torch.tensor(
            feats['res_plddt'] > plddt_threshold).int()

    def __getitem__(self, idx):
        data = self.json[idx]
        processed_data = {}
        for chain in ['target', 'binder']:
            processed_data[chain] = self.process_json_item(data[chain])
            if self._dataset_cfg.add_plddt_mask:
                self._add_plddt_mask(processed_data[chain], self._dataset_cfg.min_plddt_threshold)
            else:
                processed_data[chain]['plddt_mask'] = torch.ones_like(processed_data[chain]['res_mask'])

        # combine all chains
        combined_data = {}
        # random flip
        target = 'target'
        binder = 'binder'
        binder_seq_len = random.choice(data[binder]['binder_seq_len_range'])
        for key in processed_data['target'].keys():
            if key in ['pdb_name', 'seq_len']:
                continue
            elif key in ['res_plddt']:
                combined_data[key] = np.concatenate([processed_data[target][key], 
                                                     processed_data[binder][key][:binder_seq_len]], axis=0)
            elif key in ['rotmats_1', 'trans_1']:
                combined_data[key] = torch.cat([processed_data[target][key],  
                                                processed_data[binder][key][:binder_seq_len]], dim=0)
            elif key in ['res_mask', 'plddt_mask']:
                combined_data[key] = torch.cat([processed_data[target][key], 
                                                processed_data[binder][key][:binder_seq_len]], dim=0)
            elif key in ['aatypes_1']:
                combined_data[key] = torch.cat([processed_data[target][key], 
                                                processed_data[binder][key][:binder_seq_len]], dim=0)
            elif key in ['chain_idx']:
                combined_data[key] = np.concatenate([processed_data[target][key] + self._dataset_cfg.chain_idx_offset * 0, 
                                                     processed_data[binder][key][:binder_seq_len] + self._dataset_cfg.chain_idx_offset * 1], axis=0)
            elif key in ['res_idx']:
                combined_data[key] = np.concatenate([processed_data[target][key] + self._dataset_cfg.res_idx_offset * 0, 
                                                     processed_data[binder][key][:binder_seq_len] + self._dataset_cfg.res_idx_offset * 1], axis=0)
            elif key in ['segment_idx']:
                combined_data[key] = np.concatenate([processed_data[target][key] + self._dataset_cfg.chain_idx_offset * 0, 
                                                     processed_data[binder][key][:binder_seq_len] + self._dataset_cfg.chain_idx_offset * 1], axis=0)
            elif key in ['hotspot']:
                combined_data[key] = torch.cat([
                    processed_data[target][key],
                    torch.zeros_like(processed_data[binder][key][:binder_seq_len])  # No hotspot for binder
                ], dim=0)
            else:
                raise ValueError(f'Unrecognized key {key} for combining chains.')

        if self._dataset_cfg.noise_target:
            combined_data['diffuse_mask'] = torch.cat([
                torch.ones_like(processed_data[target]['res_mask']).bool(),
                torch.ones_like(processed_data[binder]['res_mask'][:binder_seq_len]).bool()
            ], dim=0)
        else:
            combined_data['diffuse_mask'] = torch.cat([
                torch.zeros_like(processed_data[target]['res_mask']).bool(),
                torch.ones_like(processed_data[binder]['res_mask'][:binder_seq_len]).bool()
            ], dim=0)
        combined_data['diffuse_mask'] = combined_data['diffuse_mask'].int()

        combined_data['target_seq_len'] = len(processed_data[target]['aatypes_1']) # data[target]['modeled_seq_len']
        combined_data['binder_seq_len'] = binder_seq_len

        # Storing the csv index is helpful for debugging.
        combined_data['json_idx'] = torch.ones(1, dtype=torch.long) * idx

        unique_chains, counts = np.unique(processed_data[target]['segment_idx'], return_counts=True)
        combined_data['target_seg_lens'] = counts
        target_hotspot_ranges = mask_to_ranges(processed_data[target]['hotspot'].numpy(), counts.tolist(), one_based=True)
        combined_data['target_hotspot_str'] = ranges_to_str(target_hotspot_ranges)
        combined_data['binder_hotspot_str'] = ""

        combined_data['target_pdb_name'] = data[target]['pdb_name']
        combined_data['binder_pdb_name'] = data[binder]['pdb_name']

        return combined_data

