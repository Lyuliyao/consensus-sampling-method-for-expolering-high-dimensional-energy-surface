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
    print(dim)
    # p.load_state_dict(torch.load("./model_save/potential"+str(id-1)+".pt"))
    traintime = 50000
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
    print(X_save.shape)
    data_index = data_index+1  
    while data_index<data_size:
        print('./data_save/data'+str(data_index+1)+'.npz')
        data = np.load('./data_save/data'+str(data_index+1)+'.npz')
        X_tmp = torch.tensor(data["X"].T,dtype=torch.float).to(device)
        F_tmp = torch.tensor(data["F"].T,dtype=torch.float).to(device)
        data_index = data_index+1  
        X_save = torch.cat((X_save,X_tmp),dim=1);
        F_save = torch.cat((F_save,F_tmp),dim=1);
    start_id=0
    print(X_save.shape)
    for i in range(traintime):
        optimizer.zero_grad()
        # print(X_save.size())
        if X_save.size(dim=1)>1000:
            if start_id+1000>X_save.size(dim=1):
                start_id = 0;
            X = X_save[:,start_id:start_id+1000]
            F = F_save[:,start_id:start_id+1000]
            start_id = start_id+1000
        else:
            X = X_save
            F = F_save
#        print(X.shape)
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
        print("""#!/bin/bash --login
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:v100s:1
#SBATCH --time=4:00:00
#SBATCH --mem=24G
#SBATCH -A multiscaleml
#SBATCH --job=train1

ml purge
ml Conda
conda activate pytorch_a100
python3 -c "import train as tr; import model as model ;p = tr.train({},{})"
sacct --format=JobID,Submit,Partition,CPUTime,Elapsed,MaxRSS,NodeList,ReqCPUS --units=M -j $SLURM_JOBID
            """.format(dim,id_num),file=f)
    
    
