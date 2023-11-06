import torch
import numpy as np
import torch.optim as optim
from math import *
from model import Potential
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--id', type = int, default = 20)
parser.add_argument('--dim', type = int, default = 20)

args = parser.parse_args()
id = args.id;
dim = args.dim;
device = "cuda:0"
potential1 = Potential(2*dim,10);
potential1.to(device)
def read_file(i,dim):
    index=1000
    data = np.loadtxt("../"+str(i)+"th/esamble_1/out"+str(index)+".txt")
    X    = torch.tensor(data[:,:dim],dtype=torch.float32).to(device)
    Ene    = torch.tensor(data[:,dim],dtype=torch.float32).to(device)

    while index < 50000:
        index = index + 1000;
        data = np.loadtxt("../"+str(i)+"th/esamble_1/out"+str(index)+".txt")
        X = torch.cat((X,torch.tensor(data[:,:dim],dtype=torch.float32).to(device)))
        Ene    = torch.cat((Ene,torch.tensor(data[:,dim],dtype=torch.float32).to(device)))
    return X,Ene 
    
id_now = id;
X,Ene = read_file(id_now,dim)
id_now = id_now -1;
while id_now >0:
    X_tmp,Ene_tmp =     read_file(id_now,dim)
    X = torch.cat((X,X_tmp));
    Ene = torch.cat((Ene,Ene_tmp)); 
    id_now = id_now-1;
    
traintime = 10000
potential1.to(device);
beta = 0.1
error_save=[]
loss_save =[]
optimizer = optim.Adam([
                {'params': potential1.parameters()}
            ],lr =1e-3)
for i in range(traintime):
    optimizer.zero_grad()
    losses = torch.sum((potential1.bias_potential(X.T)-Ene)**2)/X.size()[0]
    # X_Test = pi*(2*torch.rand(10000,2)-1).to(device);
    # losses = losses - 1/(torch.sum(potential1.p_bias_potential(X_Test.T,beta))/X_Test.size()[0])
    
    losses.backward()
    optimizer.step() 
    if i%100==1:
        print("i= ",i)
        print("loss1 =",(losses).detach())


potential1.to("cpu");
p1 = torch.jit.trace(potential1,torch.randn(2,100))
p1.save("potential"+str(id)+".jlt");
torch.save(potential1.state_dict(), "potential"+str(id)+".pt")