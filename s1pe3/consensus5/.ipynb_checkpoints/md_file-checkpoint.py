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
    ome2: TORSION ATOMS=8,11,30,31
    phi2: TORSION ATOMS=11,30,31,34
    psi2: TORSION ATOMS=30,31,34,53
    ome3: TORSION ATOMS=31,34,53,54
    phi3: TORSION ATOMS=34,53,54,57
    psi3: TORSION ATOMS=53,54,57,76
    ene: ENERGY
    ADACHANGE_F5 ARG=ome1,phi1,psi1,ome2,phi2,psi2,ome3,phi3,psi3 N_ESAMBLE=64 KAPPA=500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0,500.0  LABEL=restraint  BETA=1 ET=5000 FDT=25000 EPSILON=1 ALPHA=0.1 GAMMA=0.1 CPT=5000 BP=../model_save/potential{}.jlt DR=1.0 SB=0.0 AT_FILE=../AT.txt
    """.format(id-1),file=f)
    
def generate_at_file(X):
    np.savetxt('AT.txt', X.reshape(-1), delimiter='',fmt='%1.4f')
    return 0 
     
def generate_slurm_md2(nesm,dim,id):
    with open("md{}".format(str(id)),"w") as f:
        print("""#!/bin/bash
## FILENAME:  myjobsubmissionfile
#
#SBATCH -A mth210005       # allocation name
#SBATCH --nodes=4        # Total # of nodes 
#SBATCH --ntasks=512    # Total # of MPI tasks
#SBATCH --time=4:00:00   # Total run time limit (hh:mm:ss)
#SBATCH -J myjobname     # Job name
#SBATCH -p wholenode     # Queue (partition) name
#SBATCH --mail-user=useremailaddress
#SBATCH--mail-type=all   # Send email to above address at begin and end of job

# Manage processing environment, load compilers, and applications.
ml restore pp
conda activate plumed_pytorch

. /anvil/projects/x-mth210005/Liyao/pytorch/torch/sourceme.sh 
. /anvil/projects/x-mth210005/Liyao/plumed/plumed2/sourceme.sh
. /anvil/projects/x-mth210005/Liyao/plumed/gromacslib/bin/GMXRC.bash
python3 -c "import md_file as mdf;mdf.generate_plumed_file({});"
gmx_mpi grompp -o chi3_ref.tpr -c conf.gro -f grompp.mdp -maxwarn 3
gmx_mpi mdrun -s chi3_ref.tpr  -plumed ./plumed1.dat -ntomp 1 -nsteps 100  -c conf.gro -nb cpu
gmx_mpi grompp -o chi3_ref.tpr -c conf.gro -f grompp.mdp -maxwarn 3
python3 -c "import post_processing as post;import numpy as np;X = np.loadtxt('./COLVAR');X = X[-{}:,-{}:];post.generate_at_file(X)"
srun -n  512 gmx_mpi mdrun -s ../chi3_ref.tpr -c conf.gro -plumed ../plumed.dat -multidir ./esamble_? ./esamble_??  -ntomp 1 -nsteps 5000000  -nb cpu 
rm -R ./esamble_*/#*
cp ./esamble_1/conf.gro ./
cp ./esamble_1/mean.txt ./data_save/mean{}.txt
python3 -c "import post_processing as post;import numpy as np;X,F = post.collect_file({});np.savez('data_save/data{}.npz',X=X,F=F);"
sbatch train{}
sacct --format=JobID,Submit,Partition,CPUTime,Elapsed,MaxRSS,NodeList,ReqCPUS --units=M -j $SLURM_JOBID
            """.format(str(id),str(nesm),str(dim),str(id),str(dim),str(id),str(id)),file=f)
