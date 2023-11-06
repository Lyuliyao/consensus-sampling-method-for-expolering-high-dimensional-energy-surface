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

class Potential(torch.nn.Module):
    def __init__(self,N_in,N_w):
        super(Potential, self).__init__()
        self.W1 = torch.nn.Parameter(torch.randn(N_w,N_in))
        self.b1 = torch.nn.Parameter(torch.randn(N_w,1))
        self.W2 = torch.nn.Parameter(torch.randn(N_w,N_w))
        self.b2 = torch.nn.Parameter(torch.randn(N_w,1)) 
        self.W3 = torch.nn.Parameter(torch.randn(N_w,N_w))
        self.b3 = torch.nn.Parameter(torch.randn(N_w,1))        
        self.W4 = torch.nn.Parameter(torch.randn(1,N_w))
        self.b4 = torch.nn.Parameter(torch.randn(1,1))
        self.b5 = torch.nn.Parameter(torch.randn(1,1))

        self.ac = torch.nn.ELU()
    
    def forward(self, input):
        y0 = torch.cat((torch.cos(input),torch.sin(input)))
        y1 = self.W1@y0 + self.b1;
        y1 = self.ac(y1);
        y2 = self.W2@y1 + self.b2;
        y2 = self.ac(y2);
        y3 = self.W3@y2 + self.b3;
        y3 = self.ac(y3);
        output = self.W4@y3 + self.b4;
        return output
    

    def bias_potential(self, input,s):
        output = self.forward(input);
        return output*s

    def F(self, X):
        device = self.W1.device
        inputs=torch.tensor(X,dtype=torch.float).to(device).requires_grad_(True)
        inputs.to(device)
        Ene_bias = self.forward(inputs)
        v= torch.ones(Ene_bias.shape,device = device)
        F_bias = torch.autograd.grad(Ene_bias, inputs,grad_outputs=v, create_graph=True, only_inputs=True)[0]
        return F_bias

    def F2(self, inputs):
        device = self.W1.device
        inputs.requires_grad_(True)
        Ene_bias = self.forward(inputs)
        v= torch.ones(Ene_bias.shape,device = device)
        F_bias = torch.autograd.grad(Ene_bias, inputs,grad_outputs=v, create_graph=True, only_inputs=True)[0]
        return F_bias
def train(dim,id):
    p = Potential(2*dim,128);
    # p.load_state_dict(torch.load("./model_save/potential"+str(id-1)+".pt"))
    traintime = 100000
    device = "cuda:0"
    p.to(device);
    beta = 0.1
    error_save=[]
    loss_save =[]
    optimizer = optim.Adam([
                    {'params': p.parameters()}
                ],lr =1e-3)
# #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.9)
    file_list = glob.glob('./data_save/data*');
    data_size = len(file_list);
    data_index = 0;
    data = np.load(file_list[data_index])
    X_save = torch.tensor(data["X"].T,dtype=torch.float).to(device)
    F_save = torch.tensor(data["F"].T,dtype=torch.float).to(device)
    data_index = data_index+1  
    while data_index<data_size:
        data = np.load(file_list[data_index])
        X_tmp = torch.tensor(data["X"].T,dtype=torch.float).to(device)
        F_tmp = torch.tensor(data["F"].T,dtype=torch.float).to(device)
        data_index = data_index+1  
        X_save = torch.cat((X_save,X_tmp),dim=1);
        F_save = torch.cat((F_save,F_tmp),dim=1);
    for i in range(traintime):
        optimizer.zero_grad()
        # print(X_save.size())
        if X_save.size(dim=0)>500000:
            indics =torch.randperm(X_save.size(dim=0))
            indics = indics[:500000];
            X = X_save[:,indics]
            F = F_save[:,indics]
        else:
            X = X_save
            F = F_save
        F_bias = p.F2(X)
        losses = torch.sum((F_bias-F)**2)/X.size()[0]/X.size()[1]
        losses.backward()
        optimizer.step() 
#         scheduler.step()
        if i%100==1:
            print("i= ",i)
            print("loss1 =",(losses).detach())
    potential = p;
    potential.to("cpu");
    p1 = torch.jit.trace(potential,torch.randn(dim,10))
    p1.save("./model_save/potentialtest.jlt");
    torch.save(potential.state_dict(), "./model_save/potentialtest.pt") 
    return p 

