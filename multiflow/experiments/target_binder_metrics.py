import os
import gc
import jax
import numpy as np
import MDAnalysis as mda

from Bio import SeqIO
from colabdesign import mk_afdesign_model, clear_mem
from MDAnalysis.analysis import align


def cleanup_after_sample(model=None):
    """Best-effort cleanup for CPU + GPU memory after one sample."""
    if model is not None:
        try:
            model.restart()
        except Exception:
            pass

    try:
        del model
    except Exception:
        pass

    gc.collect()

    try:
        jax.clear_caches()
    except Exception:
        pass


def calculate_all_metrics(model,
                          binder_seq,
                          gt_target_binder_pdb,
                          hotspot,
                          target_chain='A',
                          binder_chain='B',
                          num_recycles=None):
    """
    Calculate all metrics for the binder prediction.
    Based on multiflow/scripts/eval_all_metrics_target_binder.py::calculate_all_metrics_for_item
    """
    results = {}

    binder_len = len(binder_seq)

    if hotspot == '':
        raise ValueError("Hotspot should not be empty for binder design evaluation.")

    model.prep_inputs(
        pdb_filename=gt_target_binder_pdb,
        target_chain=target_chain,
        binder_chain=binder_chain,
        binder_len=binder_len,
        use_binder_template=True,
        hotspot=hotspot if hotspot != '' else None,
    )
    model.set_seq(binder_seq)
    model.predict(num_recycles=num_recycles)

    # Save predicted PDB to a temp path for scRMSD calculation
    predicted_pdb_dir = os.path.dirname(gt_target_binder_pdb)
    predicted_pdb_path = os.path.join(predicted_pdb_dir, '_tmp_predicted.pdb')
    model.save_pdb(predicted_pdb_path)

    aux = getattr(model, "aux", {}) or {}
    log = aux.get("log", {}) if isinstance(aux, dict) else {}
    ipae = log.get('i_pae', 'N/A') * 31 if log.get('i_pae') is not None else 'N/A'
    binder_plddt = log.get('plddt', 'N/A') * 100 if log.get('plddt') is not None else 'N/A'

    # Recalculate ipae and binder_plddt
    Lt = model._target_len
    Lb = model._binder_len
    recalculated_ipae = np.mean(aux['pae'][:Lt, Lt:Lt+Lb] + aux['pae'][Lt:Lt+Lb, :Lt].transpose()) / 2
    recalculated_binder_plddt = np.mean(aux['plddt'][Lt:Lt+Lb]) * 100.0

    # binder_scrmsd calculation with MDAnalysis
    u1 = mda.Universe(predicted_pdb_path)
    u2 = mda.Universe(gt_target_binder_pdb)
    chain1 = u1.segments[-1].atoms.select_atoms('name CA')
    chain2 = u2.segments[-1].atoms.select_atoms('name CA')
    old_rmsd, new_rmsd = align.alignto(chain2, chain1, select="name CA", match_atoms=True)
    binder_scrmsd = new_rmsd

    results['ipae'] = ipae
    results['binder_plddt'] = binder_plddt
    results['binder_scrmsd'] = binder_scrmsd

    return results
