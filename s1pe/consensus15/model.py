import subprocess
import numpy as np
import math
import time
import logging
import os
import torch

class Potential(torch.nn.Module):
    def __init__(self,N_in,N_w):
        super(Potential, self).__init__()
        self.W1 = torch.nn.Parameter(torch.randn(N_w,N_in)/5)
        self.b1 = torch.nn.Parameter(torch.randn(N_w,1)/5)
        self.W2 = torch.nn.Parameter(torch.randn(N_w,N_w)/5)
        self.b2 = torch.nn.Parameter(torch.randn(N_w,1)/5) 
        self.W3 = torch.nn.Parameter(torch.randn(N_w,N_w)/5)
        self.b3 = torch.nn.Parameter(torch.randn(N_w,1)/5)        
        self.W4 = torch.nn.Parameter(torch.randn(1,N_w)/5)
        self.b4 = torch.nn.Parameter(torch.randn(1,1))
        self.b5 = torch.nn.Parameter(torch.randn(1,1))

        self.ac = torch.nn.Tanh()
    
    def forward(self, input):
        y0 = torch.cat((torch.cos(input),torch.sin(input)))
        y1 = self.W1@y0 + self.b1;
        y1 = self.ac(y1);
        y2 = self.W2@y1 + self.b2;
        y2 = self.ac(y2)+y1;
        y3 = self.W3@y2 + self.b3;
        y3 = self.ac(y3)+y2;
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

def init_model(dim):
    potential = Potential(2*dim,64);
    return potential


def save_model(p,dim,id):
    potential = p;
    potential.to("cpu");
    p1 = torch.jit.trace(potential,torch.randn(dim,10))
    p1.save("./model_save/potential"+str(id)+".jlt");
    torch.save(potential.state_dict(), "./model_save/potential"+str(id)+".pt")
