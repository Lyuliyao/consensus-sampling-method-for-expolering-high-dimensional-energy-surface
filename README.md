# CONSENSUS-BASED ADAPTIVE SAMPLING AND APPROXIMATION FOR HIGH-DIMENSIONAL ENERGY LANDSCAPES

This repository provides code and data for reproducing the numerical results in:

- Title: `CONSENSUS-BASED ADAPTIVE SAMPLING AND APPROXIMATION FOR HIGH-DIMENSIONAL ENERGY LANDSCAPES`
- Authors: `Liyao Lyu`, `Huan Lei`

##  Repository content

- `code/`: custom PLUMED C++ actions used by this work:
  - `Adachange_F6.cpp`
  - `Adachange_F11.cpp`
- `ala2/`: Ala2 experiments and plotting notebooks (main-text Fig. 3 and related tables).
- `s1pe/`: s1pe experiments and plotting notebooks (main-text Fig. 4 and appendix Fig. 7-9).
- `s1pe3/`: (s1pe)3 experiments (main-text Fig. 5 and appendix Fig. 10).
- `ala16/`: Ala16 experiments and plotting notebooks (main-text Fig. 6 and appendix Fig. 11).
- `REPRODUCIBILITY_CHECKLIST.md`: file-level reproducibility map.

##  Required software (recommended versions)

| Component | Recommended version/setup | Notes |
|---|---|---|
| GROMACS | `2020.7` patched with PLUMED | Matches the main `consensus*` run scripts. |
| PLUMED | `2.8.1` Torch-enabled custom build | Must support `BIAS_TORCH` and copy the `ADACHANGE_F6/F11` in src/bias. |
| LibTorch / PyTorch | `LibTorch 2.0.0` (C++), or self-built PyTorch with matching ABI/toolchain | Keep PLUMED link-time and runtime Torch stacks consistent to avoid `libc10.so` / ABI errors. |

Reference links:

- PyTorch install (official): https://pytorch.org/get-started/locally/
- PyTorch from source (official): https://github.com/pytorch/pytorch#from-source
- PLUMED installation with LibTorch linking (official): https://www.plumed.org/doc-v2.10/user-doc/html/_installation.html
- PLUMED `PYTORCH` tutorial example: https://www.plumed-tutorials.org/lessons/23/004/data/markdown/Ex-2.html

## Plumed plug-in code `F6` and `F11`

The two custom actions correspond to two model forms in the paper:

- `ADACHANGE_F6`: overdamped formulation (position-only stochastic update).
- `ADACHANGE_F11`: underdamped formulation (velocity + position update with inertia).

Code-level difference:

- `code/Adachange_F6.cpp`: updates `X_CV_local` directly.
- `code/Adachange_F11.cpp`: updates both CV position and velocity terms (underdamped dynamics with inertia).

## Reproducibility workflow

All `run_me`/`submit`/`md*`/`train*` scripts are HPC/Slurm templates.  
Before running, replace machine-specific settings for your cluster:

- `#SBATCH` account/partition/qos/constraint/resources
- environment setup (`module restore`, `conda activate`, `source .../sourceme*.sh`)
- GROMACS executable path (`gmx_mpi` location)

### 1) Generate iteration scripts (`md*`, `train*`) with `consensus.py`

In each CAS directory, run `consensus.py` first.  
It generates Slurm scripts for each CAS iteration:

- `md1 ... md(N_iter-1)` (MD sampling stage; includes the local short stage + ensemble exploration stage)
- `train1 ... train(N_iter-1)` (model update stage)

Example (s1pe):

```bash
python3 consensus.py --N_iter 21 --dim 3 --nesm 20 --clean True
```

Main commands used in this repo:

```bash
# s1pe
python3 consensus.py --N_iter 21  --dim 3  --nesm 20 --clean True
# s1pe3
python3 consensus.py --N_iter 40  --dim 9  --nesm 64 --clean True
# ala2
python3 consensus.py --N_iter 20  --dim 2  --nesm 10 --clean True
# ala16
python3 consensus.py --N_iter 241 --dim 30 --nesm 64 --clean True
```

### 2) Submit CAS in strict order

Use Slurm dependencies so each iteration is ordered as:

```text
md(i) -> train(i) -> md(i+1) -> train(i+1) -> ...
```

Then use the submit entry points:

```bash
cd s1pe/consensus15   && bash submit
cd s1pe3/consensus5   && bash submit
cd ala2/consensus11   && bash submit
cd ala16/consensus31  && bash submit
```

### 3) Post-processing (projected `.npz` surfaces)

After CAS training, run notebooks to regenerate projected surfaces, for example:

- `s1pe/consensus15/figure.ipynb`
- `s1pe3/consensus5/figure.ipynb`
- `ala16/consensus31/post_process.ipynb`
