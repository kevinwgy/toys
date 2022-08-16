#include<cmath>
#include<vector>
#include<cassert>
#include"read_vector.h"
using namespace std;

//-------------------------------------
// Problem-specific data 
//-------------------------------------
struct ProblemData {
  int D;

  vector<double> theta;
  vector<double> A;
  vector<double> phi_htm;
  vector<double> init_lambda;
  vector<double> cobb_douglas;
  vector<double> keynes;
  vector<double> factor;

  vector<double> B; //!< DxD
  vector<double> Omega_re; //!< DxD
};


//-------------------------------------
// Set problem data
//-------------------------------------
void SetProblemData(ProblemData &data)
{
  int D0;

  data.D = ReadVectorFromFile("Data/theta.txt", data.theta);

  D0 = ReadVectorFromFile("Data/A.txt", data.A);
  assert(D0==data.D);

  D0 = ReadVectorFromFile("Data/phi_htm.txt", data.phi_htm);
  assert(D0==data.D);

  D0 = ReadVectorFromFile("Data/init_lambda.txt", data.init_lambda);
  assert(D0==data.D);

  D0 = ReadVectorFromFile("Data/cobb_douglas.txt", data.cobb_douglas);
  assert(D0==data.D);

  D0 = ReadVectorFromFile("Data/keynes.txt", data.keynes);
  assert(D0==data.D);

  D0 = ReadVectorFromFile("Data/factor.txt", data.factor);
  assert(D0==data.D);

  D0 = ReadVectorFromFile("Data/B.txt", data.B);
  assert(D0==data.D*data.D);

  D0 = ReadVectorFromFile("Data/Omega_re.txt", data.Omega_re);
  assert(D0==data.D*data.D);

}

//-------------------------------------
// Set initial guess
//-------------------------------------
void SetInitialGuess(Vec X, ProblemData &data)
{
  double *x;
  VecGetArray(X,&x);

  // Set initial guess
  for (int k=0; k<data.D; k++) {  
    x[k] = 1.0;
    x[data.D+k] = data.init_lambda[k];
  }
  VecRestoreArray(X,&x);
}

//-------------------------------------
// Get number of variables & equations
//-------------------------------------
int GetNumberOfVariables(ProblemData &data)
{return 2*data.D;}

//-------------------------------------
// Evaluation function F = Fun(X)
//-------------------------------------
PetscErrorCode Fun(SNES snes, Vec X, Vec F, void *ctx)
{

  const double *x;
  double *f;
  VecGetArrayRead(X,&x);
  VecGetArray(F,&f);

  int n1, n2;
  VecGetSize(X, &n1);
  VecGetSize(F, &n2);

  ProblemData *data = (ProblemData*) ctx;
  assert(data);
  assert(n1==2*data->D);
  assert(n2==2*data->D);

  int D = data->D;
  vector<double> &theta(data->theta);
  vector<double> &A(data->A);
  vector<double> &phi_htm(data->phi_htm);
  vector<double> &init_lambda(data->init_lambda);
  vector<double> &cobb_douglas(data->cobb_douglas);
  vector<double> &keynes(data->keynes);
  vector<double> &factor(data->factor);
  vector<double> &B(data->B);
  vector<double> &Omega_re(data->Omega_re);

 //-----------------------------
 // Compute function
 
  // p(2:199, 1, 333)
  double temp = 0.0;
  //double N = 66.0;
  ///double D = 334.0;
  for(int k=0; k<2*D; k++)
    f[k]=0;

  for(int k=0; k<D; k++){

    if(cobb_douglas[k]==0){ 
      temp = 0.0;
      for(int j=0; j<D; j++)
        temp = temp + pow(B[k*D+j],theta[k])*Omega_re[k*D+j]*pow(x[j],1-theta[k]);
      f[k] = x[k] - 1/A[k]*pow(temp,1/(1-theta[k]));
    }
    else if(cobb_douglas[k]==1){
      temp = 0.0;
      for(int j=0; j<D; j++)
        temp = temp + B[k*D+j]*Omega_re[k*D+j]*log(x[j]);
      f[k] = x[k] - exp(-log(A[k])+temp);
    }
  }
  
// lambda (1-331, 334)
  for(int k=0; k<D; k++){
    if(factor[k]<2){
      temp=0;
      for(int j=0; j<D; j++){
        if(factor[j]>0)
          temp = temp + x[D+j]*pow(B[j*D+k],theta[j])*Omega_re[j*D+k]*pow(x[k],(1-theta[j]))
                       *pow(x[j],-(1-theta[j]))*pow(A[j],(theta[j]-1));        
      }
      f[D+k] = x[D+k] - temp;
    }
  }

// lambda(332)
  temp = 0;
  for(int j=0; j<D; j++){
    if(factor[j]==0)
      temp = temp + init_lambda[j]*(1-phi_htm[j])*(1-x[D+j]/x[j]/init_lambda[j]);
  }
  f[2*D-3] = temp*x[2*D-1];

// lambda(333)
  temp=0; 
  for(int j=0; j<D; j++){
    if(factor[j]==0)
      temp = temp + x[D+j];
  }
  f[2*D-2] = x[2*D-2] - (temp - x[2*D-3]);

// p(200-331)
  for(int k=0; k<D; k++){
    if(keynes[k] == 0)
      f[k] = x[k] - x[D+k]/(A[k]*init_lambda[k]);
  }

// p(334) = 1
  f[D] = x[D] - 1;

 //----------------------------

  VecRestoreArrayRead(X,&x);
  VecRestoreArray(F,&f);
  return 0;
}
                                                                               
