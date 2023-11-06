import subprocess
import numpy as np

def generate_plumed_file(id):
    with open("plumed.dat","w") as f:
        print("""

    # vim:ft=plumed
    # since simulation run in subdir, we refer to a file in the parent dir:
    ome1: TORSION ATOMS=6,1,7,8
    phi1: TORSION ATOMS=1,7,8,11
    psi1: TORSION ATOMS=7,8,11,30
    ene: ENERGY
    ADACHANGE_F11 ARG=ome1,phi1,psi1 N_ESAMBLE=20 KAPPA=500,500,500 LABEL=restraint XI=0.9 BETAL=10 ET=500 FDT=10000 EPSILON=1 ALPHA=0.1 GAMMA=1 CPT=5000 BP=../model_save/potential{}.jlt DR=1.0 SB=0.0 BETAH=2 AT_FILE=../AT.txt
    """.format(id-1),file=f)
       
def generate_plumed_file1(id):
    with open("plumed1.dat","w") as f:
        print("""

    # vim:ft=plumed
    # since simulation run in subdir, we refer to a file in the parent dir:
    ome1: TORSION ATOMS=6,1,7,8
    phi1: TORSION ATOMS=1,7,8,11
    psi1: TORSION ATOMS=7,8,11,30
    ene: ENERGY
    BIAS_TORCH ARG=ome1,phi1,psi1 BP=./model_save/potential{}.jlt  SC=1.0
    PRINT ARG=ome1,phi1,psi1 FILE=COLVAR STRIDE=1

    """.format(id-1),file=f)
def generate_at_file(X):
    np.savetxt('AT.txt', X.reshape(-1), delimiter='',fmt='%1.4f')
    return 0 
     
def generate_slurm_md2(nesm,dim,id):
    with open("md{}".format(id),"w") as f:
        print("""#!/bin/bash
## FILENAME:  myjobsubmissionfile
#
#SBATCH -A multiscaleml      # allocation name
#SBATCH --nodes=1       # Total # of nodes 
#SBATCH -n 20             # Total # of nodes 
#SBATCH --time=4:00:00        # Total run time limit (hh:mm:ss)
#SBATCH --constraint=amd20
#SBATCH --mem-per-cpu=4G
module restore plumed2_torch
conda activate plumed2torch
source  /mnt/research/MultiscaleML_group/Liyao/plumed/sourceme4.sh 
source /mnt/home/lyuliyao/gromacs_2020/bin/GMXRC.bash
export GMX_MAXBACKUP=-1
srun -n 1 /mnt/home/lyuliyao/gromacs_2020/bin/gmx_mpi grompp -o chi1.tpr -c conf.gro -f grompp2.mdp -maxwarn 3
python3 -c "import md_file as mdf;mdf.generate_plumed_file1({})"
srun -n 1 /mnt/home/lyuliyao/gromacs_2020/bin/gmx_mpi mdrun  -s chi1.tpr -plumed plumed1.dat -ntomp 1 -nsteps 10000  -c conf.gro -nb cpu
python3 -c "import post_processing as post;import numpy as np;X = np.loadtxt('./COLVAR');X = X[-{}:,-{}:];post.generate_at_file(X)"
srun -n 1 /mnt/home/lyuliyao/gromacs_2020/bin/gmx_mpi grompp -o chi1.tpr -c conf.gro -f grompp.mdp -maxwarn 3
python3 -c "import md_file as mdf;mdf.generate_plumed_file({})"
srun -n  20  /mnt/home/lyuliyao/gromacs_2020/bin/gmx_mpi mdrun -s ../chi1.tpr -c conf.gro -plumed ../plumed.dat -multidir ./esamble_? ./esamble_?? -ntomp 1 -nsteps 500000  -nb cpu 
rm -R ./esamble_*/#*
cp ./esamble_1/conf.gro ./
python3 -c "import post_processing as post;import numpy as np; X,F = post.collect_file({}); np.savez('data_save/data{}.npz',X=X,F=F);"
cp ./esamble_1/mean.txt ./data_save/mean{}.txt
sacct --format=JobID,Submit,Partition,CPUTime,Elapsed,MaxRSS,NodeList,ReqCPUS --units=M -j $SLURM_JOBID
            """.format(str(id),str(nesm),str(dim),str(id),str(dim),str(id),str(id)),file=f)
