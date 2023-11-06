import subprocess
import numpy as np

def generate_plumed_file(id):
    with open("plumed.dat","w") as f:
        print("""

    # vim:ft=plumed
    # since simulation run in subdir, we refer to a file in the parent dir:
    phi1: TORSION ATOMS=5,7,9,15
    phi2: TORSION ATOMS=15,17,19,25
    phi3: TORSION ATOMS=25,27,29,35
    phi4: TORSION ATOMS=35,37,39,45
    phi5: TORSION ATOMS=45,47,49,55
    phi6: TORSION ATOMS=55,57,59,65
    phi7: TORSION ATOMS=65,67,69,75
    phi8: TORSION ATOMS=75,77,79,85
    phi9: TORSION ATOMS=85,87,89,95
    phi10: TORSION ATOMS=95,97,99,105
    phi11: TORSION ATOMS=105,107,109,115
    phi12: TORSION ATOMS=115,117,119,125
    phi13: TORSION ATOMS=125,127,129,135
    phi14: TORSION ATOMS=135,137,139,145
    phi15: TORSION ATOMS=145,147,149,155
    psi1: TORSION ATOMS=7,9,15,17
    psi2: TORSION ATOMS=17,19,25,27
    psi3: TORSION ATOMS=27,29,35,37
    psi4: TORSION ATOMS=37,39,45,47
    psi5: TORSION ATOMS=47,49,55,57
    psi6: TORSION ATOMS=57,59,65,67
    psi7: TORSION ATOMS=67,69,75,77
    psi8: TORSION ATOMS=77,79,85,87
    psi9: TORSION ATOMS=87,89,95,97
    psi10: TORSION ATOMS=97,99,105,107
    psi11: TORSION ATOMS=107,109,115,117
    psi12: TORSION ATOMS=117,119,125,127
    psi13: TORSION ATOMS=127,129,135,137
    psi14: TORSION ATOMS=137,139,145,147
    psi15: TORSION ATOMS=147,149,155,157
    ene: ENERGY
    ADACHANGE_F11 ARG=phi1,phi2,phi3,phi4,phi5,phi6,phi7,phi8,phi9,phi10,phi11,phi12,phi13,phi14,phi15,psi1,psi2,psi3,psi4,psi5,psi6,psi7,psi8,psi9,psi10,psi11,psi12,psi13,psi14,psi15 N_ESAMBLE=64 KAPPA=500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500,500 LABEL=restraint XI=1.2 BETAL=20 ET=500 FDT=50000 EPSILON=1 ALPHA=0.1 GAMMA=1 CPT=5000 BP=../model_save/potential{}.jlt DR=1.0 SB=0.0 BETAH=2 AT_FILE=../AT.txt
    """.format(id-1),file=f)
        

def generate_plumed_file1(id):
    with open("plumed.dat","w") as f:
        print("""

    # vim:ft=plumed
    # since simulation run in subdir, we refer to a file in the parent dir:
    phi1: TORSION ATOMS=5,7,9,15
    phi2: TORSION ATOMS=15,17,19,25
    phi3: TORSION ATOMS=25,27,29,35
    phi4: TORSION ATOMS=35,37,39,45
    phi5: TORSION ATOMS=45,47,49,55
    phi6: TORSION ATOMS=55,57,59,65
    phi7: TORSION ATOMS=65,67,69,75
    phi8: TORSION ATOMS=75,77,79,85
    phi9: TORSION ATOMS=85,87,89,95
    phi10: TORSION ATOMS=95,97,99,105
    phi11: TORSION ATOMS=105,107,109,115
    phi12: TORSION ATOMS=115,117,119,125
    phi13: TORSION ATOMS=125,127,129,135
    phi14: TORSION ATOMS=135,137,139,145
    phi15: TORSION ATOMS=145,147,149,155
    psi1: TORSION ATOMS=7,9,15,17
    psi2: TORSION ATOMS=17,19,25,27
    psi3: TORSION ATOMS=27,29,35,37
    psi4: TORSION ATOMS=37,39,45,47
    psi5: TORSION ATOMS=47,49,55,57
    psi6: TORSION ATOMS=57,59,65,67
    psi7: TORSION ATOMS=67,69,75,77
    psi8: TORSION ATOMS=77,79,85,87
    psi9: TORSION ATOMS=87,89,95,97
    psi10: TORSION ATOMS=97,99,105,107
    psi11: TORSION ATOMS=107,109,115,117
    psi12: TORSION ATOMS=117,119,125,127
    psi13: TORSION ATOMS=127,129,135,137
    psi14: TORSION ATOMS=137,139,145,147
    psi15: TORSION ATOMS=147,149,155,157
    BIAS_TORCH ARG=phi1,phi2,phi3,phi4,phi5,phi6,phi7,phi8,phi9,phi10,phi11,phi12,phi13,phi14,phi15,psi1,psi2,psi3,psi4,psi5,psi6,psi7,psi8,psi9,psi10,psi11,psi12,psi13,psi14,psi15 BP=./model_save/potential{}.jlt SC=1.0
    PRINT      ARG=phi1,phi2,phi3,phi4,phi5,phi6,phi7,phi8,phi9,phi10,phi11,phi12,phi13,phi14,phi15,psi1,psi2,psi3,psi4,psi5,psi6,psi7,psi8,psi9,psi10,psi11,psi12,psi13,psi14,psi15 FILE=COLVAR STRIDE=1
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
#SBATCH --nodes=1-8          # Total # of nodes 
#SBATCH -n 512            # Total # of nodes 
#SBATCH --time=4:00:00        # Total run time limit (hh:mm:ss)
#SBATCH --constraint=amd20
#SBATCH --mem-per-cpu=4G
#SBATCH --qos=scavenger
#SBATCH --job=md1
module restore plumed2_torch
conda activate plumed2torch
source  /mnt/research/MultiscaleML_group/Liyao/plumed/sourceme6.sh 
source /mnt/home/lyuliyao/gromacs_2020/bin/GMXRC.bash
export GMX_MAXBACKUP=-1
srun -N 1 -n 1 /mnt/home/lyuliyao/gromacs_2020/bin/gmx_mpi grompp -o ala16.tpr -c conf.gro -f grompp2.mdp -maxwarn 3
python3 -c "import md_file as mdf;mdf.generate_plumed_file1({})"
srun -N 1 -n 1 /mnt/home/lyuliyao/gromacs_2020/bin/gmx_mpi mdrun -s ala16.tpr -plumed ./plumed1.dat -ntomp 1 -nsteps 10000  -c conf.gro -nb cpu
srun -N 1 -n 1 /mnt/home/lyuliyao/gromacs_2020/bin/gmx_mpi grompp -o ala16.tpr -c conf.gro -f grompp.mdp -maxwarn 3
python3 -c "import post_processing as post;import numpy as np;X = np.loadtxt('./COLVAR');X = X[-{}:,-{}:];post.generate_at_file(X)"
python3 -c "import md_file as mdf;mdf.generate_plumed_file({})"
srun -n  512  /mnt/home/lyuliyao/gromacs_2020/bin/gmx_mpi mdrun -s ../ala16.tpr -c conf.gro -plumed ../plumed.dat -multidir ./esamble_? ./esamble_??  -ntomp 1 -nsteps 2000000  -nb cpu 
cp ./esamble_1/conf.gro ./
python3 -c "import post_processing as post;import numpy as np; X,F = post.collect_file({}); np.savez('data_save/data{}.npz',X=X,F=F);"
cp ./esamble_1/mean.txt ./data_save/mean{}.txt
rm -R #*
rm -R bck.*
sacct --format=JobID,Submit,Partition,CPUTime,Elapsed,MaxRSS,NodeList,ReqCPUS --units=M -j $SLURM_JOBID
            """.format(str(id),str(nesm),str(dim),str(id),str(dim),str(id),str(id)),file=f)
