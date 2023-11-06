import torch
import subprocess
import numpy as np
import torch.optim as optim
from math import *
import model as model
import md_file as mdf
import post_processing as post
import os.path
import argparse
import glob
import io
import train as train
parser = argparse.ArgumentParser()
parser.add_argument('--N_iter', type = int, default = 20)
parser.add_argument('--dim', type = int, default = 20)
parser.add_argument('--nesm', type = int, default = 20)
parser.add_argument('--clean', type = bool, default = 20)

args = parser.parse_args()
N_iter = args.N_iter;
dim = args.dim;
nesm = args.nesm
if (args.clean==True):
    if os.path.isdir("model_save"):
        subprocess.run("rm -rf model_save",shell=True)
    if os.path.isdir("data_save"):
        subprocess.run("rm -rf data_save",shell=True)


subprocess.run("mkdir -p model_save;mkdir -p data_save",shell=True)
for i in range(nesm):
    subprocess.run("mkdir esamble_{}".format(i+1),shell=True);
def generate_at_file(X):
    np.savetxt("AT.txt",X.reshape(-1),fmt='%10.4f');

X = np.loadtxt("./COLVAR");
X = X[-nesm:,-dim:].T;
generate_at_file(X)
potential = model.init_model(dim);
model.save_model(potential,dim,0)  
for id in range(1,N_iter):
    # mdf.generate_plumed_file(id,X.reshape(-1));
    mdf.generate_slurm_md2(nesm,dim,id);
    # subprocess.run("sbatch -W md",shell=True)
    # X,F = post.collect_file(dim);
    # np.savez("data_save/data"+str(id)+".npz",X=X,F=F);
    # subprocess.run("cp ./esamble_1/mean.txt ./data_save/mean{}.txt".format(id),shell=True)
    train.generate_train(dim,id);   
    # subprocess.run("sbatch -W train",shell=True)
    
    


