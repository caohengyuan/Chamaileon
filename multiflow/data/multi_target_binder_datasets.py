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


class PDBMultiTargetBinderDataset(Dataset):
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
        print('NOTICE: filter items with target modeled_seq_len + binder modeled_seq_len > 512')
        metadata_json = []
        for x in self.raw_json:
            flag = True
            for target_i, binder_i in zip(x['targets'], x['binders']):
                if target_i['modeled_seq_len'] + binder_i['modeled_seq_len'] > 512:
                    flag = False
                    break
            if flag:
                metadata_json.append(x)
        metadata_json = sorted(metadata_json, key=lambda x: max(target_i['modeled_seq_len'] + binder_i['modeled_seq_len'] for target_i, binder_i in zip(x['targets'], x['binders'])), reverse=True)

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
            for target_i, binder_i in zip(item['targets'], item['binders']):
                target_i.update({'cluster': cluster_lookup(target_i['pdb_name'])})
                binder_i.update({'cluster': cluster_lookup(binder_i['pdb_name'])})

        self._create_split(metadata_json)

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
                eval_lengths = np.array([max(item['binders'][i]['modeled_seq_len'] for i in range(len(item['binders']))) for item in data_json])
            else:
                eval_lengths = np.array([max(item['binders'][i]['modeled_seq_len'] for i in range(len(item['binders']))) for item in data_json if all(item['binders'][i]['modeled_seq_len'] <= self._dataset_cfg.max_eval_length for i in range(len(item['binders'])))])
            all_lengths = np.sort(np.unique(eval_lengths))
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.dataset_cfg.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_json = [item for item in data_json if any(item['binders'][i]['modeled_seq_len'] in eval_lengths for i in range(len(item['binders'])))]

            # Group by 'chain_2.modeled_seq_len' in list of dicts and sample

            # Build mapping: seq_len -> list of items
            seq_len_to_items = defaultdict(list)
            for item in eval_json:
                seq_len = max(item['binders'][i]['modeled_seq_len'] for i in range(len(item['binders'])))
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
            sampled_items.sort(key=lambda x: max(x['binders'][i]['modeled_seq_len'] for i in range(len(x['binders']))), reverse=True)
            self.json = sampled_items
            self._log.info(
                f'Validation: {len(self.json)} examples with lengths {eval_lengths}')
        for idx, item in enumerate(self.json):
            item.update({'index': idx})
            if not self.is_training:
                for binder_i in item['binders']:
                    print(f"VALIDATION item {idx}: {binder_i['processed_path']}")

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
                'hotspot': torch.tensor(processed_feats['hotspot_mask']).int(),
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
        processed_item['hotspot_str'] = data.get('hotspot', None)
        aatypes_1 = du.to_numpy(processed_item['aatypes_1'])
        # TODO: why filtering chains with only one kind of amino acid?
        # if len(set(aatypes_1)) == 1:
        #     raise ValueError(f'Example {path} has only one amino acid.')
        if use_cache:
            self._cache[path] = processed_item
        return processed_item

    def _add_plddt_mask(self, feats, plddt_threshold):
        feats['plddt_mask'] = torch.tensor(
            feats['res_plddt'] > plddt_threshold).int()

    def __getitem__(self, idx):
        data = self.json[idx]
        processed_data = {
            "targets": [],
            "binders": [],
        }
        for target_i, binder_i in zip(data['targets'], data['binders']):
            processed_data['targets'].append(self.process_json_item(target_i))
            processed_data['binders'].append(self.process_json_item(binder_i))
            if self._dataset_cfg.add_plddt_mask:
                self._add_plddt_mask(processed_data['targets'][-1], self._dataset_cfg.min_plddt_threshold)
                self._add_plddt_mask(processed_data['binders'][-1], self._dataset_cfg.min_plddt_threshold)
            else:
                processed_data['targets'][-1]['plddt_mask'] = torch.ones_like(processed_data['targets'][-1]['res_mask'])
                processed_data['binders'][-1]['plddt_mask'] = torch.ones_like(processed_data['binders'][-1]['res_mask'])

        # combine all chains
        combined_data = {
            'res_plddt': [],
            'rotmats_1': [],
            'trans_1': [],
            'res_mask': [],
            'plddt_mask': [],
            'hotspot': [],
            'aatypes_1': [],
            'chain_idx': [],
            'res_idx': [],
            'hotspot_str': [],
        }
        for target_i, binder_i in zip(processed_data['targets'], processed_data['binders']):
            for key in combined_data.keys():
                if key in ['res_plddt']:
                    combined_data[key].append(np.concatenate([target_i[key], binder_i[key]], axis=0))
                elif key in ['rotmats_1', 'trans_1', 'res_mask', 'plddt_mask', 'aatypes_1']:
                    combined_data[key].append(torch.cat([target_i[key], binder_i[key]], dim=0))
                elif key in ['chain_idx']:
                    combined_data[key].append(np.concatenate([target_i[key] + self._dataset_cfg.chain_idx_offset * 0,
                                                              binder_i[key] + self._dataset_cfg.chain_idx_offset * 1], axis=0))
                elif key in ['res_idx']:
                    combined_data[key].append(np.concatenate([target_i[key] + self._dataset_cfg.res_idx_offset * 0,
                                                              binder_i[key] + self._dataset_cfg.res_idx_offset * 1], axis=0))
                elif key in ['hotspot']:
                    combined_data[key].append(torch.cat([
                        target_i[key],
                        torch.zeros_like(binder_i[key])
                    ], dim=0))
                elif key in ['hotspot_str']:
                    if target_i['hotspot_str'] is None:
                        ranges = mask_to_ranges(target_i['hotspot'].numpy(), one_based=True)
                        target_i['hotspot_str'] = ranges_to_str(ranges)
                    if binder_i['hotspot_str'] is None:
                        ranges = mask_to_ranges(binder_i['hotspot'].numpy(), one_based=True)
                        binder_i['hotspot_str'] = ranges_to_str(ranges)
                    combined_data[key].append([target_i['hotspot_str'], binder_i['hotspot_str']])
                else:
                    raise ValueError(f'Unrecognized key {key} for combining chains.')
                
        combined_data['diffuse_mask'] = []
        combined_data['target_seq_len'] = []
        combined_data['binder_seq_len'] = []
        for i, (target_i, binder_i) in enumerate(zip(processed_data['targets'], processed_data['binders'])):
            if self._dataset_cfg.noise_target:
                combined_data['diffuse_mask'].append(
                    torch.cat([
                        torch.ones_like(target_i['res_mask']).bool(),
                        torch.ones_like(binder_i['res_mask']).bool()
                    ], dim=0).int()
                )
            else:
                combined_data['diffuse_mask'].append(
                    torch.cat([
                        torch.zeros_like(target_i['res_mask']).bool(),
                        torch.ones_like(binder_i['res_mask']).bool()
                    ], dim=0).int()
                )
            combined_data['target_seq_len'].append(data['targets'][i]['modeled_seq_len'])
            combined_data['binder_seq_len'].append(data['binders'][i]['modeled_seq_len'])

        # Storing the csv index is helpful for debugging.
        combined_data['json_idx'] = torch.ones(1, dtype=torch.long) * idx

        return combined_data
