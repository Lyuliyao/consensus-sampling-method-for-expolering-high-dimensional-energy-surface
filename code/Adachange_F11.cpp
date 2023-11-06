/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2022 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "./bias/Bias.h"
#include "./bias/ActionRegister.h"
#include "tools/Random.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include <iostream>
#include <fstream>
#include "tools/Matrix.h"

#include <torch/torch.h>
#include <torch/script.h>
// convert LibTorch version to string
//#define STRINGIFY(x) #x
//#define TOSTR(x) STRINGIFY(x)
//#define LIBTORCH_VERSION TO_STR(TORCH_VERSION_MAJOR) "." TO_STR(TORCH_VERSION_MINOR) "." TO_STR(TORCH_VERSION_PATCH)

namespace PLMD {
namespace bias{
//+PLUMEDOC BIAS RESTRAINT
/*
Adds harmonic and/or linear restraints on one or more variables.

Either or both
of SLOPE and KAPPA must be present to specify the linear and harmonic force constants
respectively.  The resulting potential is given by:
\f[
  \sum_i \frac{k_i}{2} (x_i-a_i)^2 + m_i*(x_i-a_i)
\f].

The number of components for any vector of force constants must be equal to the number
of arguments to the action.

Additional material and examples can be also found in the tutorial \ref lugano-2

\par Examples

The following input tells plumed to restrain the distance between atoms 3 and 5
and the distance between atoms 2 and 4, at different equilibrium
values, and to print the energy of the restraint
\plumedfile
DISTANCE ATOMS=3,5 LABEL=d1
DISTANCE ATOMS=2,4 LABEL=d2
RESTRAINT ARG=d1,d2 AT=1.0,1.5 KAPPA=150.0,150.0 LABEL=restraint
PRINT ARG=restraint.bias
\endplumedfile

*/
//+ENDPLUMEDOC

class Adachange_F11 : public Bias {
  std::vector<float> at;
  std::vector<float> kappa;
  std::vector<float> slope;
  Matrix<float> X_CV,V_CV;
  std::vector<float> X_CV_local,V_CV_local;
  std::vector<float> E_CV,Var_CV;
  std::vector<float> E_bar_CV,Var_bar_CV;
  std::vector<float> at_ave;
  std::vector<float> E_bar_CV_unbias,Var_bar_CV_unbias;
  std::vector<float> MM,VM;
  std::vector<float> lb,rb;
  std::vector<float> X_init;
  Matrix<float> CV_save;


  Value* valueForce2;
  int fflag;
  int time_update;
  int time_now;
  int flag_is_file;
  int flag_run_plumed;
	int comm_sz,my_rank;
  int N_CV;
  int N_esamble;
  float eta1;
  float eta2;
  float gamma;
  float alpha;
  float scale_bias;
  int check_point_time;
  int full_dynamic_time;
  int check_point_time_now;
  int compute_time;
  int equilibrium_time;
  int flag_at_read;
  int esamble_index;
  float betah;
  float betal;
  float epsilon;
  float bias_eta1;
  float bias_eta2;
  float time_step_decay;
  float xi;
  std::vector<float> X_CV_tmp;
  torch::Tensor at_tensor;
  std::string at_file;
  std::string bias_potential_file;
  std::vector<Random> random;
  torch::jit::Module bias_potential_model;
  std::ofstream mean_file,var_file,out_file,err_file;
  int seed;
  void get_at_file();
public:
  explicit Adachange_F11(const ActionOptions&);
  void calculate() override;
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(Adachange_F11,"ADACHANGE_F11")

void Adachange_F11::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.use("ARG");
  keys.add("compulsory","SLOPE","0.0","specifies that the restraint is linear and what the values of the force constants on each of the variables are");
  keys.add("compulsory","KAPPA","0.0","specifies that the restraint is harmonic and what the values of the force constants on each of the variables are");
  keys.add("compulsory","BETAL","10","the low inverse temperature of the collective variable");
  keys.add("compulsory","BETAH","10","the high inverse temperature of the collective variable");
  keys.add("compulsory","AT_FILE","the position of the restraint");
  keys.add("compulsory","ETA1","0.9","the moving average scale of the mean average 0.9 should be a commonly used advice");
  keys.add("compulsory","ETA2","0.99","the moving average scale of the variance average 0.99 should be a commonly used advice");
  keys.add("compulsory","GAMMA","10","friction coeff of the collective variable");
  keys.add("compulsory","ALPHA","0.1","time step of the collective variable");
  keys.add("compulsory","CPT","1000","time to check point");
  keys.add("compulsory","FDT","1000","full dynamic time to average CV force");
  keys.add("compulsory","ET","100"," time equilibrium  to average CV force");
  keys.add("compulsory","EPSILON","0.001","small value to avoid the variance to be zero");
  keys.add("compulsory","SB","0.01","scaled parameter of how impossible the restrain is");
  keys.add("compulsory","RANDOM_SEED","5293818","Value of random number seed.");
  keys.add("compulsory","BP","NO","BIAS_POTENTIAL");
  keys.add("compulsory","DR","NO","timestep_decay");
  keys.add("compulsory","N_ESAMBLE","10","timestep_decay");
  keys.add("compulsory","XI","1","timestep_decay");

  keys.addOutputComponent("force2","default","the instantaneous value of the squared force due to this bias potential");
}

Adachange_F11::Adachange_F11(const ActionOptions&ao):
  PLUMED_BIAS_INIT(ao),
  at(getNumberOfArguments()),
  kappa(getNumberOfArguments(),0.0),
  slope(getNumberOfArguments(),0.0),
  E_CV(getNumberOfArguments(),0.0),
  Var_CV(getNumberOfArguments(),0.0),
  E_bar_CV(getNumberOfArguments(),0.0),
  Var_bar_CV(getNumberOfArguments(),0.0),
  X_CV_local(getNumberOfArguments(),0.0),
  V_CV_local(getNumberOfArguments(),0.0),
  at_ave(getNumberOfArguments(),0.0),
  E_bar_CV_unbias(getNumberOfArguments(),0.0),
  Var_bar_CV_unbias(getNumberOfArguments(),0.0),
  MM(getNumberOfArguments(),0.0),
  VM(getNumberOfArguments(),0.0),
  lb(getNumberOfArguments(),0.0),
  rb(getNumberOfArguments(),0.0),
  random(2)
{
  parseVector("SLOPE",slope);
  parseVector("KAPPA",kappa);
  parse("AT_FILE",at_file);
  parse("BETAH",betah);
  parse("BETAL",betal);
  parse("ETA1",eta1);
  parse("ETA2",eta2);
  parse("GAMMA",gamma);
  parse("ALPHA",alpha);
  parse("CPT",check_point_time);
  parse("FDT",full_dynamic_time);
  parse("EPSILON",epsilon);
  parse("SB",scale_bias);
  parse("DR",time_step_decay);
  parse("BP",bias_potential_file);
  parse("ET",equilibrium_time);
  parse("RANDOM_SEED",seed);
  parse("N_ESAMBLE",N_esamble);
  parse("XI",xi);

  if(comm.Get_rank()==0){
  comm_sz = multi_sim_comm.Get_size();
  my_rank = multi_sim_comm.Get_rank();
  }
  comm.Bcast(comm_sz,0);
  comm.Bcast(my_rank,0);
  random[0].setSeed(seed+my_rank);
  random[1].setSeed(seed+my_rank);
  checkRead();
  std::cout << my_rank << std::endl;
  N_CV = getNumberOfArguments();
  X_CV_tmp.resize(N_CV*comm_sz);
  X_init.resize(N_CV*N_esamble);

  if(comm.Get_rank()==0)
  if (my_rank==0){
    X_CV.resize(N_esamble, N_CV);
    CV_save.resize(N_esamble, N_CV);

    V_CV.resize(N_esamble, N_CV);
    get_at_file();
    esamble_index=0;
    int index = 0 ;
    for (int i = 0; i < comm_sz; ++i)
      for (int j = 0 ; j< N_CV ;  ++j){
      X_CV_tmp[index] = X_CV[i][j];
      // std::cout << index << " " << my_rank;
      index = index +1;
    }
  fflag = 1;
  mean_file.open ("mean.txt");
  for (int j = 0 ; j< N_CV ;  ++j)
  mean_file << "Mean " << j << " " ;
  mean_file << std::endl;
  mean_file.close();
  var_file.open ("var.txt");
  for (int j = 0 ; j< N_CV ;  ++j)
  var_file << "Var " << j << " ";
  var_file << std::endl;
  var_file.close();
  }
  if (bias_potential_file=="NO"){
    log.printf(" without bias potential file");
  }
  else{
    bias_potential_model = torch::jit::load(bias_potential_file);
  }
  if(comm.Get_rank()==0){
  multi_sim_comm.Bcast(X_CV_tmp,0);
  for (int j=0; j< N_CV ; j++)
  at[j] = X_CV_tmp[my_rank*N_CV+j];
  for (int i=0; i< N_CV ; i++)
  X_CV_local[i] = at[i];
  }
  comm.Bcast(at,0);
  log.printf("  at");
  for(unsigned i=0; i<at.size(); i++) log.printf(" %f",at[i]);
  log.printf("\n");
  log.printf("  with harmonic force constant");
  for(unsigned i=0; i<kappa.size(); i++) log.printf(" %f",kappa[i]);
  log.printf("\n");
  log.printf("  and linear force constant");
  for(unsigned i=0; i<slope.size(); i++) log.printf(" %f",slope[i]);
  log.printf("\n");

  addComponent("force2");
  componentIsNotPeriodic("force2");
  valueForce2=getPntrToComponent("force2");
  time_now = 0; check_point_time_now=0;compute_time=0;
  if(comm.Get_rank()==0){
  std::stringstream fileNameStream;
  fileNameStream << "out" << time_now <<  ".txt";
  std::string fileName = fileNameStream.str();
  out_file.open (fileName);
  err_file.open ("err.txt");
  }
  bias_eta1 = eta1;
  bias_eta2 = eta2;
  flag_is_file = 0;
  flag_run_plumed = 0;
}


void Adachange_F11::get_at_file(){
  std::ifstream f1;
  f1.open(at_file.c_str());  
  if (f1.good())
    for (int i=0; i < N_esamble; i++){
      for (int j=0; j<N_CV ; j++){
        float data;
        f1 >> data;
        X_CV[i][j] = data;
        E_CV[j] = data;
      }
    }
}

void Adachange_F11::calculate() {
  if (fflag){
    for (int j = 0; j<N_CV; j++){
      err_file << std::setw(10) << at[j] << " ";
    }
    fflag =0;
  }
  float ene=0.0;
  int index;
  float totf2=0.0;
  for(int i=0; i<N_CV; ++i) {
    const float cv=difference(i,at[i],getArgument(i));
    float k=kappa[i];
    const float m=slope[i];
    if (time_now<equilibrium_time/2){
      const float f = -(0.5+1*time_now/equilibrium_time)*(k*cv+m);
      ene+=0.5*k*cv*cv+m*cv;
      setOutputForce(i,f);
      totf2+=f*f;
    }
    else{
      const float f =-(k*cv+m);
      ene+=0.5*k*cv*cv+m*cv;
      setOutputForce(i,f);
      totf2+=f*f;
    }

    if (time_now>equilibrium_time){
    at_ave[i] +=  cv/(full_dynamic_time-equilibrium_time);
    }
  }
  // std::cout << "I am  commrank " << comm.Get_rank() <<  " rank" << my_rank << "in position 0, the time_now is " << time_now << std::endl;

  setBias(ene);
  valueForce2->set(totf2);

  time_now = time_now+1;

  if(comm.Get_rank()==0){
  if (time_now%full_dynamic_time==0){
    fflag =1;
    // std::cout << "I am  commrank " << comm.Get_rank() <<  " rank" << my_rank << "in position 2, the compute_time is " << compute_time << std::endl;
    time_now=0;
    compute_time+=1;
    std::vector<float> F_local(N_CV,0.0);
    for(int i=0; i<N_CV; ++i){ 
    F_local[i] =  -kappa[i]*(at_ave[i]);
    }
    // std::cout << "I am  commrank " << comm.Get_rank() <<  " rank" << my_rank << "in position 3, the compute_time is " << compute_time << std::endl;

    for (int j = 0; j<N_CV; j++){
      out_file << std::setw(10) << at[j] << " ";
    }
    for (int j = 0; j<N_CV; j++){
      out_file << std::setw(10) << F_local[j] << " ";
    }
    // std::cout << "I am  commrank " << comm.Get_rank() <<  " rank" << my_rank << "in position 4, the compute_time is " << compute_time << std::endl;

    out_file  << std::endl;
    auto input1 = torch::from_blob(at.data(), {N_CV,1}).requires_grad_(true);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input1);
    auto output1 = bias_potential_model(inputs);//.toTensor().item().toFloat();
    auto grad_output1 = torch::ones_like(output1.toTensor());
    auto F_bias = torch::autograd::grad({output1.toTensor()}, {input1}, /*grad_outputs=*/{grad_output1}, /*create_graph=*/true);
    // std::cout << "I am  commrank " << comm.Get_rank() <<  " rank" << my_rank << "in position 5, the compute_time is " << compute_time << std::endl;
    
    float err = 0 ;
    float value = 0;
    for(int i=0; i<N_CV; ++i){ 
//     std::cout << "I am  commrank " << comm.Get_rank() <<  " rank" << my_rank << "variable " << i << "error " <<err << std::endl;
    err += (F_local[i]-F_bias[0][i].item().toFloat())*(F_local[i]-F_bias[0][i].item().toFloat());
    float valuevalue = (F_local[i])*(F_local[i]);
    if (valuevalue>25){
        value += valuevalue;
    }
    else{
        value += 25;
    }
    }  
    float error_save = err;
    err = -sqrt(err)/(1+pow(sqrt(value),xi));

    err_file << std::endl;
    for (int j = 0; j<N_CV; j++){
      err_file << std::setw(10) << F_local[j] << " ";
    }
    err_file << std::endl;
    for (int j = 0; j<N_CV; j++){
      err_file << std::setw(10) << F_bias[0][j].item().toFloat() << " ";
    }
    err_file << std::endl;
    err_file  << err << " ";
    err_file  << error_save << std::endl;
  
    // std::cout << "I am  rank" << my_rank << "in position 4, the error is " << err << std::endl;
    multi_sim_comm.Barrier();
    float min_err = err;
    multi_sim_comm.Min(min_err);
    err -= min_err;


    float err_exp = exp(-betal*err);
    multi_sim_comm.Sum(err_exp);

    for (int i=0; i<N_CV; ++i){
      float diff = difference(i,X_CV_local[i],E_CV[i]);
      // if (i==1)
      // std::cout << i << " " <<  X_CV_local[i] << " " << E_CV[i] << " " << diff << std::endl;
      float x_cv = E_CV[i]-diff;
      // if (i==1)
      // std::cout << x_cv << std::endl;
      E_CV[i] = exp(-betal*err)*x_cv/err_exp;
    }
    multi_sim_comm.Sum(&E_CV[0], N_CV);


    for (int i=0; i<N_CV; ++i){
      float diff = difference(i,X_CV_local[i],E_CV[i]);
      // if (i==1)
      // std::cout << i << " " <<  X_CV_local[i] << " " << E_CV[i] << " " << diff << std::endl;
      Var_CV[i] = 2*(betal+betah)*exp(-betal*err)*diff*diff/err_exp;
    }
    multi_sim_comm.Sum(&Var_CV[0], N_CV);

    for (int i=0; i<N_CV; ++i)
    at_ave[i] = 0;
    for (int i=0; i<N_CV; ++i){
    E_bar_CV[i] = eta1*E_bar_CV[i] + (1-eta1)*E_CV[i];
    Var_bar_CV[i] = eta2*Var_bar_CV[i] + (1-eta2)*Var_CV[i];
    }
    for (int i=0; i<N_CV; ++i){
    E_bar_CV_unbias[i] = E_bar_CV[i]/(1-bias_eta1);
    Var_bar_CV_unbias[i] = Var_bar_CV[i]/(1-bias_eta2);
    }
    bias_eta1 *= eta1;
    bias_eta2 *= eta2;
    for (int i=0; i<N_CV; ++i){
    MM[i] = MM[i]  + E_CV[i];
    VM[i] = VM[i]  + Var_CV[i];
    }
    if (my_rank==0){
    mean_file.open ("mean.txt",std::ios::app);
    for (int j = 0 ; j< N_CV ;  ++j)
    mean_file  << E_bar_CV_unbias[j] << " ";
    mean_file << std::endl;
    mean_file.close();
    var_file.open ("var.txt",std::ios::app);
    for (int j = 0 ; j< N_CV ;  ++j)
    var_file  << Var_bar_CV_unbias[j] << " " ;
    var_file << std::endl;
    var_file.close();
    for (int i=0; i<N_CV; ++i){
    MM[i] = 0;
    VM[i] = 0;
    }
  }
  std::vector<float> F_CV(N_CV,0.0);
  for (int j=0; j<N_CV; ++j){
    F_CV[j] =  -difference(j , E_bar_CV_unbias[j] , X_CV_local[j] )/(Var_bar_CV_unbias[j]+epsilon) -gamma*V_CV_local[j] + sqrt(2*gamma/(alpha*betah))*random[0].Gaussian();
    // std::cout << j << " " <<  E_bar_CV_unbias[j] << " " << X_CV_local[j] << " " << -difference(j , E_bar_CV_unbias[j] , X_CV_local[j] ) << std::endl;
    // std::cout << j << " " <<  F_CV[j] << std::endl;
    // std::cout << j << " " <<  V_CV_local[j] << std::endl;
  }

  for (int j=0; j<N_CV; ++j)
  V_CV_local[j] = V_CV_local[j] + alpha*F_CV[j];

  for (int j=0; j<N_CV; ++j)
  X_CV_local[j] = X_CV_local[j]  + alpha*V_CV_local[j];
  // for (int j=0; j<N_CV; ++j){
  //   X_CV_local[j] = X_CV_local[j]  - alpha*(X_CV_local[j] - E_bar_CV_unbias[j])/(Var_bar_CV_unbias[j]+epsilon)/gamma;
  //   X_CV_local[j] = X_CV_local[j]  + sqrt(2*alpha/(gamma*betah))*random[0].Gaussian();
  // }


  for (int j=0; j< N_CV ; j++)
  at[j] = X_CV_local[j];
}
}
  comm.Bcast(at,0);
}
}
}
