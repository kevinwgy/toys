#include<cmath>
#include<vector>
#include<cassert>
#include"read_vector.h"
#include<petscsnes.h>
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

  int nVar;

  Vec X0, X1, F0, F1;

  double *J;
  int *idm, *idn;

  ProblemData() : J(NULL), idm(NULL), idn(NULL) {}
  ~ProblemData() {if (J) delete J; if(idm) delete idm; if(idn) delete idn;}
};


//-------------------------------------
// Set problem data
//-------------------------------------
void SetProblemData(ProblemData &data)
{
  data.nVar = 2;

  data.J = new double[data.nVar*data.nVar];
  data.idm = new int[data.nVar];
  data.idn = new int[data.nVar];

  for(int i=0; i<data.nVar; i++) {
    data.idm[i] = i;
    data.idn[i] = i;
  }

  VecCreate(PETSC_COMM_WORLD,&data.X0);
  VecCreate(PETSC_COMM_WORLD,&data.X1);
  VecCreate(PETSC_COMM_WORLD,&data.F0);
  VecCreate(PETSC_COMM_WORLD,&data.F1);
  VecSetSizes(data.X0,PETSC_DECIDE,data.nVar);
  VecSetSizes(data.X1,PETSC_DECIDE,data.nVar);
  VecSetSizes(data.F0,PETSC_DECIDE,data.nVar);
  VecSetSizes(data.F1,PETSC_DECIDE,data.nVar);
  VecSetFromOptions(data.X0);
  VecSetFromOptions(data.X1);
  VecSetFromOptions(data.F0);
  VecSetFromOptions(data.F1);
  
}

//-------------------------------------
// Set initial guess
//-------------------------------------
void SetInitialGuess(Vec X, ProblemData &data)
{
  double *x;
  VecGetArray(X,&x);

  // Set initial guess
  x[0] = -2.0;
  x[1] = -2.0;

  VecRestoreArray(X,&x);
}

//-------------------------------------
// Set solution bounds 
//-------------------------------------
void SetSolutionBounds(Vec XL, Vec XU, ProblemData &data)
{
  double *xl, *xu;
  VecGetArray(XL,&xl);
  VecGetArray(XU,&xu);

  // Set initial guess
  xl[0] = -1.0e8;
  xl[1] = -1.0e8;
  xu[0] = -1.0;
  xu[1] = -1.0;

  VecRestoreArray(XL,&xl);
  VecRestoreArray(XU,&xu);
}


//-------------------------------------
// Get number of variables & equations
//-------------------------------------
int GetNumberOfVariables(ProblemData &data)
{return 2;}

//-------------------------------------
// Evaluation function F = Fun(X)
//-------------------------------------
PetscErrorCode Fun(Tao tao, Vec X, Vec F, void *ctx)
{

  const double *x;
  double *f;
  VecGetArrayRead(X,&x);
  VecGetArray(F,&f);

 //----------------------------
 f[0] = x[0]*x[0] - 100.0;
 f[1] = x[1]*x[1] - 10000.0;
 //----------------------------

  VecRestoreArrayRead(X,&x);
  VecRestoreArray(F,&f);
  return 0;
}

//-------------------------------------
// Evaluation jacobian function F = Fun(X)
//-------------------------------------
PetscErrorCode Jacobian(Tao tao, Vec X, Mat J, Mat Jpre, void *ctx)
{

  ProblemData *data = (ProblemData*)ctx;

  int nVar = data->nVar;


  double *x0, *x1;
  const double *f0, *f1;
  const double *x;
  VecGetArrayRead(X,&x);

  double dx = 1.0e-6;

  for(int j=0; j<nVar; j++) {

    VecGetArray(data->X0,&x0);
    for(int i=0; i<nVar; i++)
      x0[i] = x[i];
    x0[j] -= dx;
    VecRestoreArray(data->X0, &x0);
    Fun(tao, data->X0, data->F0, ctx);
    
    VecGetArray(data->X1,&x1);
    for(int i=0; i<nVar; i++)
      x1[i] = x[i];
    x1[j] += dx;
    VecRestoreArray(data->X1, &x1);
    Fun(tao, data->X1, data->F1, ctx);
    

    VecGetArrayRead(data->F0,&f0);
    VecGetArrayRead(data->F1,&f1);

    for(int i=0; i<nVar; i++)
      data->J[i*nVar+j] = (f1[i] - f0[i])/(2.0*dx); 

    VecRestoreArrayRead(data->F0,&f0);
    VecRestoreArrayRead(data->F1,&f1);
  }
/*
  data->J[0*nVar+0] = 2.0*x[0];
  data->J[0*nVar+1] = 0.0;
  data->J[1*nVar+0] = 0.0;
  data->J[1*nVar+1] = 2.0*x[1];
*/

/*
  for(int i=0; i<2; i++)
    fprintf(stderr,"x[%d] = %e.\n", i, x[i]);
  for(int i=0; i<2; i++) {
    for(int j=0; j<2; j++) {
      fprintf(stderr,"  %e", data->J[i*nVar+j]);
      fprintf(stderr,"\n");
    }
  }
*/

  MatSetValues(J, nVar, data->idm, nVar, data->idn, (double*)data->J, INSERT_VALUES);

  MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);
  VecRestoreArrayRead(X,&x);

  return 0;
}
 
