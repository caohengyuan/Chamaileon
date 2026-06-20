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

from multiflow.data import protein, residue_constants, parsers
CA_IDX = residue_constants.atom_order['CA']


def mask_to_ranges(mask, one_based=True):
    mask = np.asarray(mask).astype(bool)
    idx = np.flatnonzero(mask)  # 0-based indices where mask==True
    if idx.size == 0:
        return []
    if one_based:
        idx = idx + 1

    ranges = []
    start = prev = idx[0]
    for x in idx[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))
    return ranges

def ranges_to_str(ranges):
    parts = []
    for a, b in ranges:
        parts.append(f"{a}" if a == b else f"{a}-{b}")
    return ",".join(parts)


class PDBTargetBinderDataset(Dataset):
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
        self._processed_pdb_path_prefix = dataset_cfg.processed_pdb_path_prefix
        self.task = task
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

        # Process clusters
        self.raw_json = json.load(open(self._dataset_cfg.json_path, 'r'))
        print('NOTICE: skip filtering for demo purposes')
        metadata_json = self.raw_json
        # metadata_json = self._filter_metadata(self.raw_json)
        print('NOTICE: filter items with target modeled_seq_len + binder modeled_seq_len > 1024')
        metadata_json = [
            x for x in metadata_json
            if x['chain_1']['modeled_seq_len'] + x['chain_2']['modeled_seq_len'] <= 1024
        ]
        metadata_json = sorted(metadata_json, key=lambda x: x['chain_1']['modeled_seq_len'] + x['chain_2']['modeled_seq_len'], reverse=True)

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
            for chain in ['chain_1', 'chain_2']:
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
                eval_lengths = np.array([item['chain_2']['modeled_seq_len'] for item in data_json])
            else:
                eval_lengths = np.array([item['chain_2']['modeled_seq_len'] for item in data_json if item['chain_2']['modeled_seq_len'] <= self._dataset_cfg.max_eval_length])
            all_lengths = np.sort(np.unique(eval_lengths))
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.dataset_cfg.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_json = [item for item in data_json if item['chain_2']['modeled_seq_len'] in eval_lengths]

            # Group by 'chain_2.modeled_seq_len' in list of dicts and sample

            # Build mapping: seq_len -> list of items
            seq_len_to_items = defaultdict(list)
            for item in eval_json:
                seq_len = item['chain_2']['modeled_seq_len']
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
            sampled_items.sort(key=lambda x: x['chain_2']['modeled_seq_len'], reverse=True)
            self.json = sampled_items
            self._log.info(
                f'Validation: {len(self.json)} examples with lengths {eval_lengths}')
        for idx, item in enumerate(self.json):
            item.update({'index': idx})
            if not self.is_training:
                print(f"VALIDATION item {idx}: {item['chain_2']['processed_path']}")

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
                'res_idx': new_res_idx,
            }

    def process_json_item(self, data):
        path = os.path.join(self._processed_pdb_path_prefix, data['processed_path'])
        seq_len = data['modeled_seq_len']
        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]
        processed_item = self._process_json_item(path)
        processed_item['pdb_name'] = data['pdb_name']
        processed_item['seq_len'] = seq_len
        aatypes_1 = du.to_numpy(processed_item['aatypes_1'])
        # TODO: why filtering chains with only one kind of amino acid?
        # if len(set(aatypes_1)) == 1:
        #     raise ValueError(f'Example {path} has only one amino acid.')
        if use_cache:
            self._cache[path] = processed_item
        return processed_item
    
    def _process_json_item_2(self, processed_file_path, processed_feats):
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
                'res_idx': new_res_idx,
            }
    
    def process_json_item_2(self, data, processed_file_path, processed_feats):
        seq_len = data['modeled_seq_len']
        processed_item = self._process_json_item_2(processed_file_path, processed_feats)
        processed_item['pdb_name'] = data['pdb_name']
        processed_item['seq_len'] = seq_len
        return processed_item

    def _add_plddt_mask(self, feats, plddt_threshold):
        feats['plddt_mask'] = torch.tensor(
            feats['res_plddt'] > plddt_threshold).int()

    def __getitem__(self, idx):
        data = self.json[idx]

        # random flip
        i = random.choice([0, 1]) + 1
        target = f'chain_{i}'
        binder = f'chain_{3-i}'

        # set target in the center
        processed_path = {}
        processed_path[target] = data[target]['processed_path']
        processed_path[binder] = data[binder]['processed_path']
        processed_feats = {}
        processed_feats[target] = du.read_pkl(processed_path[target])
        processed_feats[binder] = du.read_pkl(processed_path[binder])
        target_center = np.sum(processed_feats[target]['atom_positions'][:, CA_IDX], axis=0) / (np.sum(processed_feats[target]['atom_mask'][:, CA_IDX]) + 1e-5)
        # np.sum(bb_pos, axis=0) / (np.sum(chain_feats['bb_mask']) + 1e-5)
        processed_feats[target]['atom_positions'] = processed_feats[target]['atom_positions'] - target_center
        processed_feats[binder]['atom_positions'] = processed_feats[binder]['atom_positions'] - target_center

        processed_data = {}
        for chain in ['chain_1', 'chain_2']:
            processed_data[chain] = self.process_json_item_2(data[chain], processed_path[chain], processed_feats[chain])
            # processed_data[chain] = self.process_json_item(data[chain])
            if self._dataset_cfg.add_plddt_mask:
                self._add_plddt_mask(processed_data[chain], self._dataset_cfg.min_plddt_threshold)
            else:
                processed_data[chain]['plddt_mask'] = torch.ones_like(processed_data[chain]['res_mask'])

        # combine all chains
        combined_data = {}
        for key in processed_data['chain_1'].keys():
            if key in ['pdb_name', 'seq_len']:
                continue
            elif key in ['res_plddt']:
                combined_data[key] = np.concatenate([processed_data[target][key], 
                                                     processed_data[binder][key]], axis=0)
            elif key in ['rotmats_1', 'trans_1']:
                combined_data[key] = torch.cat([processed_data[target][key],  
                                                processed_data[binder][key]], dim=0)
            elif key in ['res_mask', 'plddt_mask']:
                combined_data[key] = torch.cat([processed_data[target][key], 
                                                processed_data[binder][key]], dim=0)
            elif key in ['aatypes_1']:
                combined_data[key] = torch.cat([processed_data[target][key], 
                                                processed_data[binder][key]], dim=0)
            elif key in ['chain_idx']:
                combined_data[key] = np.concatenate([processed_data[target][key] + self._dataset_cfg.chain_idx_offset * 0, 
                                                     processed_data[binder][key] + self._dataset_cfg.chain_idx_offset * 1], axis=0)
            elif key in ['res_idx']:
                combined_data[key] = np.concatenate([processed_data[target][key] + self._dataset_cfg.res_idx_offset * 0, 
                                                     processed_data[binder][key] + self._dataset_cfg.res_idx_offset * 1], axis=0)
            elif key in ['hotspot']:
                combined_data[key] = torch.cat([
                    processed_data[target][key],
                    torch.zeros_like(processed_data[binder][key])  # No hotspot for binder
                ], dim=0)
            else:
                raise ValueError(f'Unrecognized key {key} for combining chains.')

        if self._dataset_cfg.noise_target:
            combined_data['diffuse_mask'] = torch.cat([
                torch.ones_like(processed_data[target]['res_mask']).bool(),
                torch.ones_like(processed_data[binder]['res_mask']).bool()
            ], dim=0)
        else:
            combined_data['diffuse_mask'] = torch.cat([
                torch.zeros_like(processed_data[target]['res_mask']).bool(),
                torch.ones_like(processed_data[binder]['res_mask']).bool()
            ], dim=0)
        combined_data['diffuse_mask'] = combined_data['diffuse_mask'].int()

        combined_data['target_seq_len'] = data[target]['modeled_seq_len']
        combined_data['binder_seq_len'] = data[binder]['modeled_seq_len']

        # Storing the csv index is helpful for debugging.
        combined_data['json_idx'] = torch.ones(1, dtype=torch.long) * idx

        target_hotspot_ranges = mask_to_ranges(processed_data[target]['hotspot'].numpy(), one_based=True)
        combined_data['target_hotspot_str'] = ranges_to_str(target_hotspot_ranges)
        binder_hotspot_ranges = mask_to_ranges(processed_data[binder]['hotspot'].numpy(), one_based=True)
        combined_data['binder_hotspot_str'] = ranges_to_str(binder_hotspot_ranges)

        return combined_data
