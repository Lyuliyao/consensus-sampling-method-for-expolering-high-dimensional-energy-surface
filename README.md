# Reproducibility Package for SISC Badge

This repository provides code and data for reproducing the numerical results in:

- Title: `CONSENSUS-BASED ADAPTIVE SAMPLING AND APPROXIMATION FOR HIGH-DIMENSIONAL ENERGY LANDSCAPES`
- Authors: `Liyao Lyu`, `Huan Lei`
- Purpose of this repository: reproducibility materials for the paper (code, parameters, data, and run instructions).
<!-- 
## 1. Archive metadata (fill before submission)

For SISC badge submission, complete this block in the final frozen snapshot:

| Item | Value |
|---|---|
| Public repository URL | `https://github.com/Lyuliyao/consensus-sampling-method-for-expolering-high-dimensional-energy-surface` |
| Immutable commit used for manuscript results | `32299392ad6a8e40f14bf6f683ba8cacef6e70c5` |
| Supplementary zip filename (uploaded to SISC) | `TO_FILL` |
| DOI archive (Zenodo or similar, optional but recommended) | `TO_FILL` |
| Date of frozen snapshot | `TO_FILL` | -->

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
| PLUMED | `2.8.1` Torch-enabled custom build | Must support `BIAS_TORCH` and custom `ADACHANGE_F6/F11`. |
| LibTorch / PyTorch | `LibTorch 2.0.0` (C++), or self-built PyTorch with matching ABI/toolchain | Keep PLUMED link-time and runtime Torch stacks consistent to avoid `libc10.so` / ABI errors. |

Reference links:

- PyTorch install (official): https://pytorch.org/get-started/locally/
- PyTorch from source (official): https://github.com/pytorch/pytorch#from-source
- PLUMED installation with LibTorch linking (official): https://www.plumed.org/doc-v2.10/user-doc/html/_installation.html
- PLUMED `PYTORCH` tutorial example: https://www.plumed-tutorials.org/lessons/23/004/data/markdown/Ex-2.html

## Plumed plug-in code `F6` and `F11`

The two custom actions correspond to two model forms in the paper:

- `ADACHANGE_F6`: overdamped formulation (position-only stochastic update), corresponding to Eq. (6).
- `ADACHANGE_F11`: underdamped formulation (velocity + position update with inertia), corresponding to Eq. (11).

Code-level difference:

- `code/Adachange_F6.cpp`: updates `X_CV_local` directly.
- `code/Adachange_F11.cpp`: updates `V_CV_local` and then `X_CV_local`.

##  Environment checks

Run before reproducing:

```bash
gmx --version | head -n 5        # or gmx_mpi --version | head -n 5
plumed info --version
```

If runtime fails with `libc10.so` errors, your LibTorch runtime path is inconsistent with the PLUMED/GROMACS build.

##  Reproducibility workflow

Two supported modes are provided.

### Mode A: Regenerate paper figures/tables from included data and trained checkpoints

1. Create output folder:

```bash
cd consensus-sampling-method-for-expolering-high-dimensional-energy-surface
mkdir -p figure
```

2. Execute plotting notebooks:

```bash
cd ala2
jupyter nbconvert --to notebook --execute --inplace figure.ipynb

cd ../s1pe
jupyter nbconvert --to notebook --execute --inplace figure.ipynb

cd ../ala16
jupyter nbconvert --to notebook --execute --inplace figure.ipynb
```

These notebooks generate figure files (e.g., `figure/ala2_2D.eps`, `figure/chi1_2D.eps`, `figure/ala16_2D.eps`) and print numerical errors (`l_infty`, `l_2`) used in table comparisons.

### Mode B: Full rerun of MD + CAS training + post-processing

All `run_me` scripts are HPC/Slurm templates and contain machine-specific hard-coded paths (`module restore`, `conda activate`, `source .../sourceme*.sh`, absolute GROMACS paths). Update them first.

Main CAS entry points:

```bash
cd s1pe/consensus15   && sbatch run_me
cd s1pe3/consensus5   && sbatch run_me
cd ala2/consensus11   && sbatch run_me
cd ala16/consensus31  && sbatch run_me
```

After CAS training, run post-processing notebooks to regenerate projected `.npz` surfaces (for example `s1pe/consensus15/figure.ipynb`, `s1pe3/consensus5/figure.ipynb`, `ala16/consensus31/post_process.ipynb`), then execute the plotting notebooks in Mode A.

## 7. Paper result map (item -> code/data)

| Paper item | Main paths | Reproduction route | Main output |
|---|---|---|---|
| Fig. 1 (workflow schematic) | manuscript figure source | Conceptual workflow figure in paper text | figure in manuscript |
| Fig. 2, Table 3 (1D Rastrigin) | `TO_ADD_IN_FINAL_ARCHIVE` | Add and run dedicated 1D script/notebook in final archive | adaptive points, residual plot, moment table |
| Fig. 3, Table 1 (Ala2, 2D) | `ala2/2D_2`, `ala2/ves/ves_100000000`, `ala2/consensus11`, `ala2/figure.ipynb` | Run reference/VES/CAS jobs, then execute `ala2/figure.ipynb` | `figure/ala2_2D.eps`, `figure/ala2_2D.pdf`, printed `l_infty`, `l_2` |
| Fig. 4, Fig. 7-9, Table 5 (s1pe, 3D projections) | `s1pe/consensus15`, `s1pe/example2D_6`, `s1pe/figure.ipynb` | Run CAS + reference jobs, then execute `s1pe/figure.ipynb` | `figure/chi1_2D.eps`, `figure/chi1_ome_-2.eps`, `figure/chi1_ome_-1.eps`, `figure/chi1_phi_-1.eps`, `figure/chi1_phi_0.eps`, `figure/chi1_psi_0.eps`, `figure/chi1_psi_-1.eps` |
| Fig. 5, Fig. 10 ((s1pe)3, 9D projections) | `s1pe3/consensus5`, `s1pe3/example2D2` | Run `s1pe3/consensus5/run_me`, then execute `s1pe3/consensus5/figure.ipynb` to regenerate projected `.npz` | `s1pe3/consensus5/example_2D_example*.npz` |
| Fig. 6, Fig. 11 (Ala16, 30D projections) | `ala16/consensus31`, `ala16/example_2D_12_anvil`, `ala16/example_2D_13_anvil`, `ala16/figure.ipynb` | Run CAS + reference jobs, run `ala16/consensus31/post_process.ipynb`, then execute `ala16/figure.ipynb` | `figure/ala16_2D.eps`, `figure/ala16_phi5_psi5_2D.eps`, etc. |
| Table 2 (NN architecture) | paper appendix + model definitions | Directly specified in manuscript appendix and model configs | architecture table values |
| Table 4 (iteration-wise Ala2 errors) | `TO_ADD_IN_FINAL_ARCHIVE` | Add explicit script/notebook that computes per-iteration `l_2` and `l_infty` | table values per iteration |

## 8. Known limitations to fix before final badge request

To fully satisfy strict long-term reproducibility checks, complete these items in the final archive:

- Replace absolute external RiD paths in plotting notebooks with local repository paths. Current external path usage appears in `ala2/figure.ipynb`.
- Replace absolute external RiD paths in plotting notebooks with local repository paths. Current external path usage appears in `s1pe/figure.ipynb`.
- Ensure scripts/notebooks for all remaining reported tables are explicitly included in this repository snapshot (for example, iteration-wise error tables not yet fully scripted in one command path).
- Upload a zipped immutable snapshot to SISC supplementary materials and record the exact filename and checksum in Section 1.
