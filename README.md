# Chamaileon

This is the official implementation of Chamaileon. [ICML2026 Spotlight]

## Chamaileon: Cross-Context Binder Design with Contextualized Modeling and Mixed Sampling.

by *Hengyuan Cao, Shizhuo Cheng, Mingxuan Liu, Weicheng Huang, Yunhong Lu, Chenxi Cai, Yan Zhang, Min Zhang*

## Updates

[2026-06-20] upload codebase.

## 🔍 Introduction

**<h3>Abstract</h3>**

> The rapid evolution of generative models has unlocked new potentials in protein binder design, apivotal task in structural biology, by facilitatingend-to-end generation via joint sequence-structuremodeling or hallucination. However, existing approaches are predominantly implemented under a single-target, single-state assumption, limiting their ability to model multi-target or multi-state interactions required for advanced function-oriented protein design. Here, we introduce Chamaileon, which unifies multi-target and multi-state binder design by formulating the problem as cross-context binding landscape modeling. The framework is underpinned by a training paradigm termed In-Context Complex Co-Design(I3CD) for context-aware sequence-structure co-modeling. During inference, we employ Mixture-of-Paths Sampling (MoPS), a scalable strategythat optimizes a single sequence across contextswhile alleviating the scarcity of high-quality multi-conformational paired data. Extensive evaluationon our newly constructed benchmark, CROSS, demonstrates that Chamaileon effectively generates sequences adaptable to diverse conformational landscapes and multi-target requirements.

## 🚀 Quick Start

### Dependencies
```bash
# 1. Create conda environment and install conda-only packages
mamba env create -f environment.yml
conda activate chamaileon

# 2. Install pip dependencies
pip install -r requirements.txt
pip install -r requirements-gpu.txt
pip install -e .

# 3. Set LD_LIBRARY_PATH for JAX CUDA support (add to your .bashrc or run before each session)
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
```

**Note:** JAX requires CUDA 12 cuPTI to use GPU. If you see `Unable to load cuPTI` errors, make sure the `LD_LIBRARY_PATH` above points to your CUDA 12 installation's CUPTI directory.

### Checkpoints

You can download the weights of Chamaileon at [this link](https://huggingface.co/caohy666/Chamaileon).

For beam search evaluation, AlphaFold2 Multimer parameters are also required:
```bash
# download alphafold weights
mkdir params
curl -fsSL https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar | tar x -C params
```

You can put the downloaded files and folders under the `download` folder.
```
download/
├── chamaileon.ckpt
├── config.yaml
└── AlphaFold2/
    └── params/
```

### Inference

```bash
# binder design based on a single target
bash scripts/inference_target_binder_eval.sh

# corss-context binder design
bash scripts/inference_multi_target_binder_eval.sh
```

## 📋 Citation

If you find this work useful, please cite:
```bibtex

```

## 📕 Acknowledgements

This codebase is built upon several excellent open-source projects. We gratefully acknowledge their contributions:

- **[Multiflow](https://github.com/andrew-cr/multiflow)** (MIT License) — Our codebase is built on the Multiflow framework.
- **[ColabDesign](https://github.com/sokrypton/ColabDesign)** (Apache 2.0) — Used for AlphaFold2 Multimer-based evaluation of designed binders.
- **[OpenFold](https://github.com/aqlaboratory/openfold)** (Apache 2.0) — Protein structure utilities and residue constants.
- **[ProteinMPNN](https://github.com/dauparas/ProteinMPNN)** (MIT License) — Inverse folding for sequence design.

Please also cite the original works if you use this codebase:

```bibtex
@article{campbell2024generative,
  title={Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design},
  author={Campbell, Andrew and Yim, Jason and Barzilay, Regina and Rainforth, Tom and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2402.04997},
  year={2024}
}
@article {Ahdritz2022.11.20.517210,
	author = {Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O{\textquoteright}Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolò and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed},
	title = {{O}pen{F}old: {R}etraining {A}lpha{F}old2 yields new insights into its learning mechanisms and capacity for generalization},
	elocation-id = {2022.11.20.517210},
	year = {2022},
	doi = {10.1101/2022.11.20.517210},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/10.1101/2022.11.20.517210},
	eprint = {https://www.biorxiv.org/content/early/2022/11/22/2022.11.20.517210.full.pdf},
	journal = {bioRxiv}
}
@article {Dauparas2022.06.03.494563,
	author = {Dauparas, J. and Anishchenko, I. and Bennett, N. and Bai, H. and Ragotte, R. J. and Milles, L. F. and Wicky, B. I. M. and Courbet, A. and de Haas, R. J. and Bethel, N. and Leung, P. J. Y. and Huddy, T. F. and Pellock, S. and Tischer, D. and Chan, F. and Koepnick, B. and Nguyen, H. and Kang, A. and Sankaran, B. and Bera, A. K. and King, N. P. and Baker, D.},
	title = {Robust deep learning based protein sequence design using ProteinMPNN},
	elocation-id = {2022.06.03.494563},
	year = {2022},
	doi = {10.1101/2022.06.03.494563},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {While deep learning has revolutionized protein structure prediction, almost all experimentally characterized de novo protein designs have been generated using physically based approaches such as Rosetta. Here we describe a deep learning based protein sequence design method, ProteinMPNN, with outstanding performance in both in silico and experimental tests. The amino acid sequence at different positions can be coupled between single or multiple chains, enabling application to a wide range of current protein design challenges. On native protein backbones, ProteinMPNN has a sequence recovery of 52.4\%, compared to 32.9\% for Rosetta. Incorporation of noise during training improves sequence recovery on protein structure models, and produces sequences which more robustly encode their structures as assessed using structure prediction algorithms. We demonstrate the broad utility and high accuracy of ProteinMPNN using X-ray crystallography, cryoEM and functional studies by rescuing previously failed designs, made using Rosetta or AlphaFold, of protein monomers, cyclic homo-oligomers, tetrahedral nanoparticles, and target binding proteins.One-sentence summary A deep learning based protein sequence design method is described that is widely applicable to current design challenges and shows outstanding performance in both in silico and experimental tests.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2022/06/04/2022.06.03.494563},
	eprint = {https://www.biorxiv.org/content/early/2022/06/04/2022.06.03.494563.full.pdf},
	journal = {bioRxiv}
}

```