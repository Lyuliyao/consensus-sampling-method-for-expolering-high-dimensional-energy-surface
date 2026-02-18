# Reproducibility Checklist (Repository-Specific)

This file is a practical index of where to find code, data, parameters, and run instructions in this repository.

## 1) Core Method Code

- C++ implementation files:
  - `code/Adachange_F6.cpp`
  - `code/Adachange_F11.cpp`

## 2) Experiment Inventory

| Experiment family | Main reproducibility folder | Typical run entry | Parameter files | Data/output examples |
|---|---|---|---|---|
| s1pe | `s1pe/consensus15` | `run_me` | `plumed*.dat`, `grompp*.mdp`, `chi1.tpr` | `*.npz`, `data_save/`, `traj_comp.xtc`, checkpoints |
| s1pe (2D examples) | `s1pe/example2D_6` | `run_me_*` | `plumed_*.dat`, `chi1.tpr` | `fes_*.dat`, `confout.gro`, checkpoints |
| s1pe3 | `s1pe3/consensus5` | `run_me` | local `plumed/grompp` inputs | generated trajectories and model outputs |
| ala2 | `ala2/consensus11` | `run_me` | local `plumed/grompp` inputs | `.npz`, trajectories, checkpoints |
| ala16 | `ala16/consensus31` | `run_me` | local `plumed/grompp` inputs | `.npz`, `data_save/`, trajectories, checkpoints |

## 3) Minimum Items Needed to Reproduce a Figure/Table

For each figure/table in a paper draft, make sure you can point to:

- [ ] **Run script** (e.g., `run_me`, `md*`, `train*`)
- [ ] **Model script(s)** (e.g., `model.py`, `consensus.py`, training scripts)
- [ ] **Parameter files** (`plumed*.dat`, `grompp*.mdp`, system files)
- [ ] **Input/initial states** (`.gro`, `.tpr`, checkpoints if required)
- [ ] **Post-processing script** (`post.py`, `post_processing.py`, notebooks)
- [ ] **Generated output artifact** (`.npz`, text tables, FES files, trajectories)

## 4) Suggested Author-Facing README Additions Before Submission

Before releasing a badge-ready snapshot, add:

1. paper title and author list,
2. software version pins (Python/GROMACS/PLUMED),
3. exact command lines per figure/table,
4. expected runtime and hardware notes,
5. where final immutable archived snapshot is stored.

## 5) Archival Note

This repository currently includes many generated artifacts directly in experiment folders. For long-term reproducibility releases, keep a frozen tagged snapshot and avoid moving paths after publication.
