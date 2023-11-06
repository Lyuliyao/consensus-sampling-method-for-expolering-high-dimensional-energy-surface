import subprocess
import glob
import numpy as np
def clean_file(nprocs=1):
    for i in range(nprocs):
        subprocess.run("rm -rf esamble_{}".format(i+1),shell=True);
    return 0

def collect_file(dim):
    file_list = glob.glob('./esamble_*/out*.txt')
    data = np.loadtxt(file_list[0])
    X    = np.float32(data[:,:dim])
    F    = np.float32(data[:,dim:])

    for i in range(1,len(file_list)):
        data = np.loadtxt(file_list[i])
        X = np.concatenate((X,np.float32(data[:,:dim])))
        F    = np.concatenate((F,np.float32(data[:,dim:])))
    return X,F

def generate_at_file(X):
    np.savetxt("AT.txt",X.reshape(-1),fmt='%10.4f');

def save_model(p,dim,id):
    potential = p;
    potential.to("cpu");
    p1 = torch.jit.trace(potential,torch.randn(dim,100))
    p1.save("./model_save/potential"+str(id)+".jlt");
    torch.save(potential.state_dict(), "./model_save/potential"+str(id)+".pt")