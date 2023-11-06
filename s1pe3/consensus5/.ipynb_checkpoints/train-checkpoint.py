import torch
import subprocess
import numpy as np
import torch.optim as optim
from math import *
import model as model
import md_file as mdf
import os.path
import argparse
import glob

def train(dim,id):
    p = model.init_model(dim);
    # p.load_state_dict(torch.load("./model_save/potential"+str(id-1)+".pt"))
    traintime = 200000
    device = "cuda:0"
    p.to(device);
    beta = 0.1
    error_save=[]
    loss_save =[]
    optimizer = optim.Adam([
                    {'params': p.parameters()}
                ],lr =1e-3)
    file_list = glob.glob('./data_save/data*');
    data_size = len(file_list);
    data_index = 0;
    data = np.load('./data_save/data'+str(data_index+1)+'.npz')
    X_save = torch.tensor(data["X"].T,dtype=torch.float).to(device)
    F_save = torch.tensor(data["F"].T,dtype=torch.float).to(device)
    data_index = data_index+1  
    while data_index<data_size:
        data = np.load('./data_save/data'+str(data_index+1)+'.npz')
        X_tmp = torch.tensor(data["X"].T,dtype=torch.float).to(device)
        F_tmp = torch.tensor(data["F"].T,dtype=torch.float).to(device)
        data_index = data_index+1  
        X_save = torch.cat((X_save,X_tmp),dim=1);
        F_save = torch.cat((F_save,F_tmp),dim=1);
    start_id=0
    for i in range(traintime):
        optimizer.zero_grad()
        # print(X_save.size())
        if X_save.size(dim=1)>1000:
            if start_id+1000>X_save.size(dim=1):
                start_id = 0;
                indics =torch.randperm(X_save.size(dim=1))
                X_save = X_save[:,indics]
                F_save = F_save[:,indics]
            X = X_save[:,start_id:start_id+1000]
            F = F_save[:,start_id:start_id+1000]
            start_id = start_id+1000
        else:
            X = X_save
            F = F_save
        F_bias = p.F2(X)
        losses = torch.sum((F_bias-F)**2)/torch.sum((F)**2)
        losses.backward()
        optimizer.step() 
#         scheduler.step()
        if i%100==1:
            print("i= ",i, flush=True)
            print("loss1 =",(losses).detach(), flush=True)
            
    model.save_model(p,dim,id)  
    return p 


def generate_train(dim,id_num):
    with open("train{}".format(id_num),"w") as f:
        print("""#!/bin/bash
## FILENAME:  myjobsubmissionfile
#
##SBATCH -A myallocation       # allocation name
#SBATCH --nodes=1             # Total # of nodes 
#SBATCH --ntasks-per-node=5   # Number of MPI ranks per node (one rank per GPU)
#SBATCH --gres=gpu:1          # Number of GPUs per node
#SBATCH --time=4:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J myjobname          # Job name         # Name of stderr error file
#SBATCH -p gpu                # Queue (partition) name
#SBATCH --mail-user=lyuliyao@msu.edu;
#SBATCH --mail-type=all       # Send email to above address at begin and end of job

# Manage processing environment, load compilers, and applications.

module purge
module load openmpi/4.1.1-gcc8.3.1
module load fftw/3.3.8
module load anaconda
conda activate  pytorch10 
python3 -c "import train as tr; import model as model ;p = tr.train({},{})"
sacct --format=JobID,Submit,Partition,CPUTime,Elapsed,MaxRSS,NodeList,ReqCPUS --units=M -j $SLURM_JOBID
            """.format(dim,id_num,id_num+1),file=f)
