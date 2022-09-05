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

  Vec F0, F1;

  // related to jacobian matrix
  double *J;
  int *idm, *idn;

  // a global vector that stores x
  double *xglob;
  bool xglob_ready;
  void UpdateXGlob(Vec X);

  ProblemData() : J(NULL), idm(NULL), idn(NULL), xglob(NULL), xglob_ready(false) {}
  ~ProblemData() {if (J) delete J; if(idm) delete idm; if(idn) delete idn; if(xglob) delete xglob;}
};


void ProblemData::UpdateXGlob(Vec X)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);

  const double *xloc;
  VecGetArrayRead(X,&xloc);

  int i0, imax;
  VecGetOwnershipRange(X, &i0, &imax);

  const int *ranges;
  int counts[mpi_size], displacements[mpi_size];
  VecGetOwnershipRanges(X, &ranges);
  for(int i=0; i<mpi_size; i++) {
    counts[i] = ranges[i+1] - ranges[i];
    displacements[i] = ranges[i];
  }

  assert(i0 == displacements[mpi_rank]);
  assert(imax == i0 + counts[mpi_rank]);

  for(int i=i0; i<imax; i++)
    xglob[i0] = xloc[i-i0];     

  MPI_Allgatherv(MPI_IN_PLACE, imax-i0, MPI_DOUBLE, xglob, counts, displacements, MPI_DOUBLE, PETSC_COMM_WORLD);

  VecRestoreArrayRead(X,&xloc);
}

//-------------------------------------
// Set problem data
//-------------------------------------
void SetProblemData(ProblemData &data)
{
  data.nVar = 2;

  VecCreate(PETSC_COMM_WORLD,&data.F0);
  VecCreate(PETSC_COMM_WORLD,&data.F1);
  VecSetSizes(data.F0,PETSC_DECIDE,data.nVar);
  VecSetSizes(data.F1,PETSC_DECIDE,data.nVar);
  VecSetFromOptions(data.F0);
  VecSetFromOptions(data.F1);
  

  int mpi_rank, mpi_size;
  MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);

  data.xglob = new double[data.nVar];

  const int *ranges;
  int counts[mpi_size], displacements[mpi_size];
  VecGetOwnershipRanges(data.F0, &ranges);
  for(int i=0; i<mpi_size; i++) {
    counts[i] = ranges[i+1] - ranges[i];
    displacements[i] = ranges[i];
  }

  data.J = new double[counts[mpi_rank]*data.nVar];
  data.idm = new int[counts[mpi_rank]];
  data.idn = new int[data.nVar];

  for(int i=0; i<counts[mpi_rank]; i++)
    data.idm[i] = displacements[mpi_rank]+i;

  for(int i=0; i<data.nVar; i++)
    data.idn[i] = i;

}

//-------------------------------------
// Set initial guess
//-------------------------------------
void SetInitialGuess(Vec X, ProblemData &data)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);

  double *x;
  VecGetArray(X,&x);
  int i0, imax;
  VecGetOwnershipRange(X, &i0, &imax);

  // Set initial guess
  if(!mpi_rank)
    x[0] = -2.0;
  else
    x[0] = -5.0;

  VecRestoreArray(X,&x);
}

//-------------------------------------
// Set solution bounds 
//-------------------------------------
void SetSolutionBounds(Vec XL, Vec XU, ProblemData &data)
{
  int mpi_rank, mpi_size;
  MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);

  double *xl, *xu;
  VecGetArray(XL,&xl);
  VecGetArray(XU,&xu);

  int i0, imax;
  VecGetOwnershipRange(XL, &i0, &imax);

  // Set initial guess
  if(!mpi_rank) {
    xl[0] = -1.0e8;
    xu[0] = -1.0;
  } else {
    xl[0] = -1.0e8;
    xu[0] = -1.0;
  }

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
//Note: X is used iff "data->xglob_ready == false"
PetscErrorCode Fun(Tao tao, Vec X, Vec F, void *ctx)
{

  int mpi_rank, mpi_size;
  MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);

  ProblemData *data = (ProblemData*)ctx;

  if(!data->xglob_ready)
    data->UpdateXGlob(X);

  double *f;
  VecGetArray(F,&f);

  int i0, imax;
  VecGetOwnershipRange(X, &i0, &imax);

  //----------------------------
  double *x = data->xglob;
  if(!mpi_rank)
    f[0] = x[0]*x[0] - 100.0;
  else
    f[0] = x[1]*x[1] - 10000.0;
  //----------------------------

  VecRestoreArray(F,&f);

  //VecView(F,PETSC_VIEWER_STDOUT_WORLD);
  

  data->xglob_ready = false;

  return 0;
}

//-------------------------------------
// Evaluation jacobian function F = Fun(X)
//-------------------------------------
PetscErrorCode Jacobian(Tao tao, Vec X, Mat J, Mat Jpre, void *ctx)
{

  ProblemData *data = (ProblemData*)ctx;

  data->UpdateXGlob(X);

  int nVar = data->nVar;

  const double *f0, *f1;

  double *x = data->xglob;
  double dx = 1.0e-6;

  int i0, imax, my_size;
  VecGetOwnershipRange(X, &i0, &imax);
  my_size = imax - i0;

  for(int j=0; j<nVar; j++) {

    x[j] -= dx;
    data->xglob_ready = true;
    Fun(tao, X, data->F0, ctx);

    x[j] += 2.0*dx; 
    data->xglob_ready = true;
    Fun(tao, X, data->F1, ctx);

    x[j] -= dx; //restore
 
    VecGetArrayRead(data->F0,&f0);
    VecGetArrayRead(data->F1,&f1);

    for(int i=0; i<my_size; i++)
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

  MatSetValues(J, my_size, data->idm, nVar, data->idn, (double*)data->J, INSERT_VALUES);

  MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);

/*
  VecView(X,PETSC_VIEWER_STDOUT_WORLD);
  MatView(J,PETSC_VIEWER_STDOUT_WORLD);
  exit(-1);
*/
  return 0;
}
 
