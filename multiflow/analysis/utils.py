import numpy as np
import os
import re
from multiflow.data import protein
from typing import Optional, Sequence, List, Union



def create_full_prot(
        atom37: np.ndarray,
        atom37_mask: np.ndarray,
        aatype=None,
        b_factors=None,
    ):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors)


def write_prot_to_pdb(
        prot_pos: np.ndarray,
        file_path: str,
        aatype: np.ndarray=None,
        overwrite=False,
        no_indexing=False,
        b_factors=None,
    ):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip('.pdb')
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max([
            int(re.findall(r'_(\d+).pdb', x)[0]) for x in existing_files if re.findall(r'_(\d+).pdb', x)
            if re.findall(r'_(\d+).pdb', x)] + [0])
    if not no_indexing:
        save_path = file_path.replace('.pdb', '') + f'_{max_existing_idx+1}.pdb'
    else:
        save_path = file_path

    if aatype is not None:
        assert aatype.ndim == prot_pos.ndim - 2

    with open(save_path, 'w') as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(
                    pos37, atom37_mask, aatype=aatype[t], b_factors=b_factors)
                pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(
                prot_pos, atom37_mask, aatype=aatype, b_factors=b_factors)
            pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f'Invalid positions shape {prot_pos.shape}')
        f.write('END')
    return save_path



# def _set_chain_id_in_pdb_text(pdb_text: str, chain_id: str) -> str:
#     if not chain_id:
#         return pdb_text
#     cid = chain_id[0]
#     out = []
#     # keepends=True 保留原有行尾（包括 \n 或 \r\n）
#     for line in pdb_text.splitlines(keepends=True):
#         # 把行尾的换行字符分离出来
#         body = line.rstrip('\r\n')
#         ending = line[len(body):]
#         if body.startswith(("ATOM  ", "HETATM", "TER")):
#             if len(body) < 22:
#                 body = body.ljust(22)
#             body = body[:21] + cid + body[22:]
#         out.append(body + ending)
#     return ''.join(out)

# def create_full_prot_multi_chain(
#         atom37: np.ndarray,
#         atom37_mask: np.ndarray,
#         aatype=None,
#         b_factors=None,
#         chain_index: Optional[np.ndarray] = None,
#         residue_index: Optional[np.ndarray] = None,
#     ):
#     """
#     构造 protein.Protein。新增参数:
#     - chain_index: (n_res,) 整数数组，指定每个残基属于哪个 chain（0,1,2,...）
#     - residue_index: 可指定残基编号数组（默认 0..n-1）
#     """
#     assert atom37.ndim == 3
#     assert atom37.shape[-1] == 3
#     assert atom37.shape[-2] == 37
#     n = atom37.shape[0]

#     if residue_index is None:
#         residue_index = np.arange(n, dtype=int)
#     if chain_index is None:
#         chain_index = np.zeros(n, dtype=int)
#     else:
#         chain_index = np.asarray(chain_index, dtype=int)
#         if chain_index.shape[0] != n:
#             raise ValueError("chain_index length must equal number of residues")

#     if b_factors is None:
#         b_factors = np.zeros([n, 37])
#     if aatype is None:
#         aatype = np.zeros(n, dtype=int)

#     return protein.Protein(
#         atom_positions=atom37,
#         atom_mask=atom37_mask,
#         aatype=aatype,
#         residue_index=residue_index,
#         chain_index=chain_index,
#         b_factors=b_factors)

# def write_prot_to_pdb_multi_chain(
#         prot_pos: np.ndarray,
#         file_path: str,
#         aatype: np.ndarray=None,
#         overwrite=False,
#         no_indexing=False,
#         b_factors=None,
#         chain_splits: Optional[Sequence[int]] = None,
#         chain_ids: Optional[Sequence[str]] = None,
#         renumber_residues_per_chain: bool = False,
#         write_separate_files: bool = False,
#     ):
#     if overwrite:
#         max_existing_idx = 0
#     else:
#         file_dir = os.path.dirname(file_path)
#         file_name = os.path.basename(file_path).strip('.pdb')
#         existing_files = [x for x in os.listdir(file_dir) if file_name in x]
#         max_existing_idx = max([
#             int(re.findall(r'_(\d+).pdb', x)[0]) for x in existing_files if re.findall(r'_(\d+).pdb', x)
#             if re.findall(r'_(\d+).pdb', x)] + [0])
#     if not no_indexing:
#         save_path = file_path.replace('.pdb', '') + f'_{max_existing_idx+1}.pdb'
#     else:
#         save_path = file_path

#     if aatype is not None:
#         assert aatype.ndim == prot_pos.ndim - 2

#     # validate chain_splits / chain_ids
#     def _validate_splits(splits, n_res):
#         if splits is None:
#             return None
#         if sum(splits) != n_res:
#             raise ValueError(f"chain_splits sum ({sum(splits)}) != number of residues ({n_res})")
#         ranges = []
#         cur = 0
#         for l in splits:
#             ranges.append((cur, cur + l))
#             cur += l
#         return ranges
    
#     # main write
#     if prot_pos.ndim == 4:
#         T = prot_pos.shape[0]
#         N_res = prot_pos.shape[1]
#         chain_ranges = _validate_splits(chain_splits, N_res)
#         # if write_separate_files True, we'll create one file per model
#         if write_separate_files:
#             # write separate files for each t
#             out_paths = []
#             for t in range(T):
#                 path_t = save_path.replace('.pdb', f'_model{t+1}.pdb')
#                 with open(path_t, 'w') as f:
#                     pos37 = prot_pos[t]
#                     if chain_ranges is None:
#                         atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
#                         prot = create_full_prot_multi_chain(pos37, atom37_mask,
#                                                 aatype=aatype[t] if aatype is not None else None,
#                                                 b_factors=b_factors)
#                         pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
#                         f.write(pdb_prot)
#                     else:
#                         # slice per chain and write, replace chain id if provided
#                         for i, (s, e) in enumerate(chain_ranges):
#                             pos_chain = pos37[s:e]
#                             atom37_mask = np.sum(np.abs(pos_chain), axis=-1) > 1e-7
#                             aatype_chain = aatype[t][s:e] if aatype is not None else None
#                             # optionally renumber residue indexes per chain
#                             if renumber_residues_per_chain:
#                                 res_idx = np.arange(1, e - s + 1, dtype=int)
#                             else:
#                                 # global numbering
#                                 res_idx = np.arange(s, e, dtype=int)
#                             prot = create_full_prot_multi_chain(pos_chain, atom37_mask, aatype=aatype_chain,
#                                                     b_factors=(b_factors[t][s:e] if (b_factors is not None and getattr(b_factors,'ndim',None)==4) else b_factors),
#                                                     residue_index=res_idx)
#                             pdb_chain = protein.to_pdb(prot, model=1, add_end=False)
#                             if chain_ids is not None:
#                                 pdb_chain = _set_chain_id_in_pdb_text(pdb_chain, chain_ids[i])
#                             f.write(pdb_chain)
#                             if i != len(chain_ranges) - 1:
#                                 f.write("TER\n")
#                     f.write("END\n")
#                 out_paths.append(path_t)
#             return out_paths  # 返回列表（每个模型路径）
#         else:
#             # 写入同一文件，用 MODEL 段分隔每个 t
#             with open(save_path, 'w') as f:
#                 for t in range(T):
#                     pos37 = prot_pos[t]
#                     if chain_ranges is None:
#                         atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
#                         prot = create_full_prot_multi_chain(pos37, atom37_mask,
#                                                 aatype=aatype[t] if aatype is not None else None,
#                                                 b_factors=b_factors)
#                         pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
#                         f.write(pdb_prot)
#                     else:
#                         # for each chain slice, write its fragment and set chain id
#                         # 把整个 model 当作若干链片段写入，model=t+1
#                         for i, (s, e) in enumerate(chain_ranges):
#                             pos_chain = pos37[s:e]
#                             atom37_mask = np.sum(np.abs(pos_chain), axis=-1) > 1e-7
#                             aatype_chain = aatype[t][s:e] if aatype is not None else None
#                             if renumber_residues_per_chain:
#                                 res_idx = np.arange(1, e - s + 1, dtype=int)
#                             else:
#                                 res_idx = np.arange(s, e, dtype=int)
#                             prot = create_full_prot_multi_chain(pos_chain, atom37_mask, aatype=aatype_chain,
#                                                     b_factors=(b_factors[t][s:e] if (b_factors is not None and getattr(b_factors,'ndim',None)==4) else b_factors),
#                                                     residue_index=res_idx)
#                             pdb_chain = protein.to_pdb(prot, model=t + 1, add_end=False)
#                             if chain_ids is not None:
#                                 pdb_chain = _set_chain_id_in_pdb_text(pdb_chain, chain_ids[i])
#                             f.write(pdb_chain)
#                             if i != len(chain_ranges) - 1:
#                                 f.write("TER\n")
#                 f.write("END\n")
#             return save_path

#     elif prot_pos.ndim == 3:
#         # single model
#         N_res = prot_pos.shape[0]
#         chain_ranges = _validate_splits(chain_splits, N_res)
#         with open(save_path, 'w') as f:
#             if chain_ranges is None:
#                 atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
#                 prot = create_full_prot_multi_chain(prot_pos, atom37_mask, aatype=aatype, b_factors=b_factors)
#                 pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
#                 f.write(pdb_prot)
#             else:
#                 for i, (s, e) in enumerate(chain_ranges):
#                     pos_chain = prot_pos[s:e]
#                     atom37_mask = np.sum(np.abs(pos_chain), axis=-1) > 1e-7
#                     aatype_chain = aatype[s:e] if aatype is not None else None
#                     if renumber_residues_per_chain:
#                         res_idx = np.arange(1, e - s + 1, dtype=int)
#                     else:
#                         res_idx = np.arange(s, e, dtype=int)
#                     prot = create_full_prot_multi_chain(pos_chain, atom37_mask, aatype=aatype_chain,
#                                             b_factors=(b_factors[s:e] if (b_factors is not None and getattr(b_factors,'ndim',None)==3) else b_factors),
#                                             residue_index=res_idx)
#                     pdb_chain = protein.to_pdb(prot, model=1, add_end=False)
#                     if chain_ids is not None:
#                         pdb_chain = _set_chain_id_in_pdb_text(pdb_chain, chain_ids[i])
#                     f.write(pdb_chain)
#                     if i != len(chain_ranges) - 1:
#                         f.write("TER\n")
#             f.write("END\n")
#         return save_path
#     else:
#         raise ValueError(f'Invalid positions shape {prot_pos.shape}')


def create_full_prot_multi_chain(
    atom37: np.ndarray,
    atom37_mask: np.ndarray,
    aatype: Optional[np.ndarray] = None,
    b_factors: Optional[np.ndarray] = None,
    chain_index: Optional[np.ndarray] = None,
    residue_index: Optional[np.ndarray] = None,
):
    """
    Construct a protein.Protein object for a (possibly multi-chain) protein.

    Parameters
    - atom37: (N_res, 37, 3) coordinates
    - atom37_mask: (N_res, 37) boolean mask of present atoms
    - aatype: (N_res,) integer residue types (optional)
    - b_factors: (N_res, 37) float b-factors per atom (optional)
    - chain_index: (N_res,) integer chain ids (0-based). If None, all zeros.
    - residue_index: (N_res,) integer residue sequence numbers to be printed in PDB (resSeq).
                     If None, defaults to 1..N_res (PDB-friendly).

    Returns:
    - protein.Protein instance
    """
    assert atom37.ndim == 3 and atom37.shape[-2] == 37 and atom37.shape[-1] == 3
    n = atom37.shape[0]

    if residue_index is None:
        # Default: 1-based residue numbering (typical in PDB)
        residue_index = np.arange(1, n + 1, dtype=int)
    else:
        residue_index = np.asarray(residue_index, dtype=int)

    if chain_index is None:
        chain_index = np.zeros(n, dtype=int)
    else:
        chain_index = np.asarray(chain_index, dtype=int)
        if chain_index.shape[0] != n:
            raise ValueError("chain_index length must equal number of residues")

    if b_factors is None:
        b_factors = np.zeros([n, 37], dtype=float)
    if aatype is None:
        aatype = np.zeros(n, dtype=int)

    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
    )


def _set_chain_id_in_pdb_text(pdb_text: str, chain_id: str) -> str:
    """
    Replace the chain id character (column 22, 1-based) in ATOM/HETATM/TER lines
    with the provided single-character chain_id for the whole pdb_text.

    This function replaces the chain id for all ATOM/HETATM/TER lines to the same
    chain_id. If you need different chains within the same model, prefer to
    generate a single protein with chain_index set and let protein.to_pdb
    emit chain ids, or use the helper `_set_chain_ids_by_resseq` below.

    Preserves original line endings by using splitlines(keepends=True).

    Parameters:
    - pdb_text: original PDB text
    - chain_id: single-character chain identifier (string); if empty or None,
                returns pdb_text unchanged.

    Returns:
    - Modified pdb text with chain id replaced.
    """
    if not chain_id:
        return pdb_text
    cid = chain_id[0]
    out_lines = []
    for line in pdb_text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        if body.startswith(("ATOM  ", "HETATM", "TER")):
            # Ensure we have at least 22 characters to safely replace index 21.
            if len(body) < 22:
                body = body.ljust(22)
            body = body[:21] + cid + body[22:]
        out_lines.append(body + ending)
    return "".join(out_lines)


def _set_chain_ids_by_resseq(pdb_text: str, resseq_to_chainid: dict) -> str:
    """
    Replace chain id based on residue sequence number (resSeq) printed in the PDB.

    The PDB column for resSeq is columns 23-26 (1-based) -> indices [22:26].
    This function parses resSeq and looks up resseq_to_chainid[resSeq] to decide
    which chain id to write on that ATOM/HETATM/TER line.

    Parameters:
    - pdb_text: original PDB text
    - resseq_to_chainid: dict mapping integer resSeq -> single-char chain id

    Returns:
    - Modified pdb text with chain ids replaced per-residue.
    """
    out_lines = []
    for line in pdb_text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        if body.startswith(("ATOM  ", "HETATM", "TER")):
            # Ensure we can safely slice up to index 26
            if len(body) < 26:
                body = body.ljust(26)
            resseq_str = body[22:26].strip()
            try:
                resseq = int(resseq_str)
            except Exception:
                out_lines.append(body + ending)
                continue
            chain_id = resseq_to_chainid.get(resseq, None)
            if chain_id:
                cid = chain_id[0]
                if len(body) < 22:
                    body = body.ljust(22)
                body = body[:21] + cid + body[22:]
        out_lines.append(body + ending)
    return "".join(out_lines)


def write_prot_to_pdb_multi_chain(
    prot_pos: np.ndarray,
    file_path: str,
    aatype: Optional[np.ndarray] = None,
    overwrite: bool = False,
    no_indexing: bool = False,
    b_factors: Optional[np.ndarray] = None,
    chain_splits: Optional[Sequence[Union[int, np.integer]]] = None,
    chain_ids: Optional[Sequence[str]] = None,
    renumber_residues_per_chain: bool = False,
    write_separate_files: bool = False,
):
    """
    Write one or multiple models (batch) to PDB. Each model may contain multiple chains.

    Behavior:
    - If prot_pos.ndim == 3: single model, shape (N_res, 37, 3)
    - If prot_pos.ndim == 4: batch of models, shape (T, N_res, 37, 3)
      by default all models are written into the same PDB using MODEL/ENDMDL,
      unless write_separate_files=True, in which case each model is written
      to a separate file (path suffix _model{t+1}.pdb).

    Multi-chain support:
    - Provide chain_splits e.g. [L1, L2] where sum(Li) == N_res to split
      the residue axis into chains.
    - Provide chain_ids e.g. ['A', 'B'] to map chain index -> chain letter.
    - The function constructs a single protein.Protein per model with
      chain_index set, then calls protein.to_pdb once per model. If the underlying
      protein.to_pdb respects chain_index, chain ids will appear correctly.
      If not, _set_chain_ids_by_resseq is used as a post-process to map residue
      numbers to chain ids.

    Returns:
    - save_path (str) if single-file behavior, or list of paths if write_separate_files=True.
    """
    # determine save_path indexing
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path) or "."
        file_name = os.path.basename(file_path).replace(".pdb", "")
        try:
            existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        except FileNotFoundError:
            existing_files = []
        idxs = []
        for x in existing_files:
            m = re.findall(r"_(\d+)\.pdb$", x)
            if m:
                try:
                    idxs.append(int(m[0]))
                except Exception:
                    pass
        max_existing_idx = max(idxs + [0])
    if not no_indexing:
        save_path = file_path.replace(".pdb", "") + f"_{max_existing_idx+1}.pdb"
    else:
        save_path = file_path

    # aatype shape check: aatype.ndim should be prot_pos.ndim - 2 (as in earlier code)
    if aatype is not None:
        assert aatype.ndim == prot_pos.ndim - 2, "aatype.ndim must equal prot_pos.ndim - 2"

    def _validate_splits_local(splits, n_res):
        """Normalize splits to a list of Python ints and return [(s,e), ...] ranges."""
        if splits is None:
            return None
        # Accept torch tensor / numpy / list / tuple / scalar
        # Convert to Python list of ints
        try:
            # lazy import torch if available
            import torch

            is_torch = True
        except Exception:
            torch = None
            is_torch = False

        if is_torch and isinstance(splits, torch.Tensor):
            splits_list = splits.to("cpu").tolist()
        elif isinstance(splits, np.ndarray):
            splits_list = splits.tolist()
        elif isinstance(splits, (list, tuple)):
            splits_list = list(splits)
        else:
            splits_list = [splits]

        splits_list = [int(x) for x in splits_list]
        n_res = int(n_res)

        if any(x <= 0 for x in splits_list):
            raise ValueError("chain_splits must contain positive integers")

        if sum(splits_list) != n_res:
            raise ValueError(
                f"chain_splits sum ({sum(splits_list)}) != number of residues ({n_res})"
            )

        ranges = []
        cur = 0
        for l in splits_list:
            ranges.append((cur, cur + l))
            cur += l
        return ranges

    # Main writing logic
    if prot_pos.ndim == 4:
        T = prot_pos.shape[0]
        N_res = prot_pos.shape[1]
        chain_ranges = _validate_splits_local(chain_splits, N_res)

        if write_separate_files:
            out_paths: List[str] = []
            for t in range(T):
                path_t = save_path.replace(".pdb", f"_model{t+1}.pdb")
                pos37 = prot_pos[t]
                # build one Protein for this model with chain_index/residue_index
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7

                # chain_index and residue_index
                if chain_ranges is None:
                    chain_idx = np.zeros(N_res, dtype=int)
                else:
                    chain_idx = np.zeros(N_res, dtype=int)
                    for cid, (s, e) in enumerate(chain_ranges):
                        chain_idx[s:e] = cid

                if renumber_residues_per_chain and chain_ranges is not None:
                    residue_idx = np.zeros(N_res, dtype=int)
                    for cid, (s, e) in enumerate(chain_ranges):
                        # per-chain numbering starts at 1
                        residue_idx[s:e] = np.arange(1, e - s + 1, dtype=int)
                else:
                    # global numbering from 1..N_res
                    residue_idx = np.arange(1, N_res + 1, dtype=int)

                # slice aatype and b_factors per model if provided
                aatype_model = aatype[t] if (aatype is not None) else None
                b_factors_model = None
                if b_factors is not None:
                    # handle common shapes: (T,N,37) or (T,N,37,?) etc.
                    try:
                        b_factors_model = b_factors[t]
                    except Exception:
                        b_factors_model = b_factors

                prot = create_full_prot_multi_chain(
                    pos37,
                    atom37_mask,
                    aatype=aatype_model,
                    b_factors=b_factors_model,
                    chain_index=chain_idx,
                    residue_index=residue_idx,
                )

                pdb_text = protein.to_pdb(prot, model=1, add_end=False)
                # If chain_ids supplied, try to enforce them per residue number
                if chain_ids is not None and chain_ranges is not None:
                    # Map the printed resSeq values to chain ids:
                    resseq_to_chain = {}
                    for cid, (s, e) in enumerate(chain_ranges):
                        # value(s) in residue_idx[s:e] are the resSeq values printed
                        for rv in residue_idx[s:e]:
                            resseq_to_chain[int(rv)] = chain_ids[cid]
                    pdb_text = _set_chain_ids_by_resseq(pdb_text, resseq_to_chain)

                # write and optionally add TER between chains if protein.to_pdb didn't
                with open(path_t, "w") as f:
                    f.write(pdb_text)
                    # Ensure there is a TER between chains if protein.to_pdb didn't include them.
                    # Many to_pdb implementations include TER; double-adding is harmless for many parsers,
                    # but adding blindly may duplicate TER lines. We skip inserting here.
                    f.write("END\n")
                out_paths.append(path_t)
            return out_paths

        else:
            # write all models into a single file using MODEL/ENDMDL
            with open(save_path, "w") as f:
                for t in range(T):
                    pos37 = prot_pos[t]
                    atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7

                    if chain_ranges is None:
                        chain_idx = np.zeros(N_res, dtype=int)
                    else:
                        chain_idx = np.zeros(N_res, dtype=int)
                        for cid, (s, e) in enumerate(chain_ranges):
                            chain_idx[s:e] = cid

                    if renumber_residues_per_chain and chain_ranges is not None:
                        residue_idx = np.zeros(N_res, dtype=int)
                        for cid, (s, e) in enumerate(chain_ranges):
                            residue_idx[s:e] = np.arange(1, e - s + 1, dtype=int)
                    else:
                        residue_idx = np.arange(1, N_res + 1, dtype=int)

                    aatype_model = aatype[t] if (aatype is not None) else None
                    b_factors_model = None
                    if b_factors is not None:
                        try:
                            b_factors_model = b_factors[t]
                        except Exception:
                            b_factors_model = b_factors

                    prot = create_full_prot_multi_chain(
                        pos37,
                        atom37_mask,
                        aatype=aatype_model,
                        b_factors=b_factors_model,
                        chain_index=chain_idx,
                        residue_index=residue_idx,
                    )

                    pdb_text = protein.to_pdb(prot, model=t + 1, add_end=False)

                    if chain_ids is not None and chain_ranges is not None:
                        # Map resSeq -> chain id and replace in text
                        resseq_to_chain = {}
                        for cid, (s, e) in enumerate(chain_ranges):
                            for rv in residue_idx[s:e]:
                                resseq_to_chain[int(rv)] = chain_ids[cid]
                        pdb_text = _set_chain_ids_by_resseq(pdb_text, resseq_to_chain)

                    f.write(pdb_text)
                f.write("END\n")
            return save_path

    elif prot_pos.ndim == 3:
        N_res = prot_pos.shape[0]
        chain_ranges = _validate_splits_local(chain_splits, N_res)

        pos37 = prot_pos
        atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7

        if chain_ranges is None:
            chain_idx = np.zeros(N_res, dtype=int)
        else:
            chain_idx = np.zeros(N_res, dtype=int)
            for cid, (s, e) in enumerate(chain_ranges):
                chain_idx[s:e] = cid

        if renumber_residues_per_chain and chain_ranges is not None:
            residue_idx = np.zeros(N_res, dtype=int)
            for cid, (s, e) in enumerate(chain_ranges):
                residue_idx[s:e] = np.arange(1, e - s + 1, dtype=int)
        else:
            residue_idx = np.arange(1, N_res + 1, dtype=int)

        aatype_model = aatype if (aatype is not None) else None
        b_factors_model = b_factors

        prot = create_full_prot_multi_chain(
            pos37,
            atom37_mask,
            aatype=aatype_model,
            b_factors=b_factors_model,
            chain_index=chain_idx,
            residue_index=residue_idx,
        )

        pdb_text = protein.to_pdb(prot, model=1, add_end=False)
        if chain_ids is not None and chain_ranges is not None:
            resseq_to_chain = {}
            for cid, (s, e) in enumerate(chain_ranges):
                for rv in residue_idx[s:e]:
                    resseq_to_chain[int(rv)] = chain_ids[cid]
            pdb_text = _set_chain_ids_by_resseq(pdb_text, resseq_to_chain)

        with open(save_path, "w") as f:
            f.write(pdb_text)
            f.write("END\n")
        return save_path
    else:
        raise ValueError(f"Invalid positions shape {prot_pos.shape}")