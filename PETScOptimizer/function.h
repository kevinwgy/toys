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
  vector<double> freq_re;
  vector<double> factor2;
  vector<double> sol_shockest;

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
    xglob[i] = xloc[i-i0];     

  MPI_Allgatherv(MPI_IN_PLACE, imax-i0, MPI_DOUBLE, xglob, counts, displacements, MPI_DOUBLE, PETSC_COMM_WORLD);
/*
  if(!mpi_rank) {
    for(int i=0; i<imax-i0; i++)
      fprintf(stderr,"xloc[%d] = %e.\n", i, xloc[i]);
  }
*/
/*
  if(!mpi_rank) {
    for(int i=0; i<3*D; i++)
      fprintf(stderr,"xglob[%d] = %e.\n", i, xglob[i]);
  }
*/
  VecRestoreArrayRead(X,&xloc);
}

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

  D0 = ReadVectorFromFile("Data/factor2.txt", data.factor2);
  assert(D0==data.D);

  D0 = ReadVectorFromFile("Data/freq_re.txt", data.freq_re);
  assert(D0==data.D);

  D0 = ReadVectorFromFile("Data/B.txt", data.B);
  assert(D0==data.D*data.D);

  D0 = ReadVectorFromFile("Data/Omega_re.txt", data.Omega_re);
  assert(D0==data.D*data.D);

  D0 = ReadVectorFromFile("Data/sol_shockest.txt", data.sol_shockest);
  assert(D0==3*data.D);

  data.nVar = 3*data.D;

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
  for (int k=0; k<3*data.D; k++) {
    if(k<i0 || k>=imax)
      continue;
    x[k-i0] = data.sol_shockest[k];
  }

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
  for(int i=0; i<imax-i0; i++) {
    xl[i] = 0.0;
    xu[i] = PETSC_INFINITY;
  }

  VecRestoreArray(XL,&xl);
  VecRestoreArray(XU,&xu);

}


//-------------------------------------
// Get number of variables & equations
//-------------------------------------
int GetNumberOfVariables(ProblemData &data)
{return data.nVar;}

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


  int D = data->D;
  vector<double> &theta(data->theta);
  vector<double> &A(data->A);
  vector<double> &phi_htm(data->phi_htm);
  vector<double> &init_lambda(data->init_lambda);
  vector<double> &cobb_douglas(data->cobb_douglas);
  vector<double> &keynes(data->keynes);
  vector<double> &factor(data->factor);
  vector<double> &factor2(data->factor2);
  vector<double> &freq_re(data->freq_re);
  vector<double> &B(data->B);
  vector<double> &Omega_re(data->Omega_re);
  vector<double> &sol_shockest(data->sol_shockest);


  //----------------------------
  // x=[mc,p,lambda]

  double temp = 0.0;
  double temp2 = 0.0;
  double temp3 = 0.0;

  double *x = data->xglob;


  for(int k=i0; k<imax; k++)
    f[k-i0]=0;

  // mc
  for(int k=i0; k<imax; k++){

    if(k>=D)
      break;

    temp = 0.0;
    if(cobb_douglas[k]==0){
      for(int j=0; j<D; j++)
        temp = temp + pow(B[k*D+j],theta[k])*Omega_re[k*D+j]*pow(x[D+j],1-theta[k]);
      f[k-i0] = x[k] - 1/A[k]*pow(temp,1/(1-theta[k]));
    }
    else if(cobb_douglas[k]==1){
      for(int j=0; j<D; j++)
        temp = temp + B[k*D+j]*Omega_re[k*D+j]*log(x[D+j]);
      f[k-i0] = x[k] - exp(-log(A[k])+temp);
    }
    else{
      f[k-i0] = x[k] - x[D+k];
    }
  }


  // lambda (1-331, 334)
  for(int i=std::max(i0, 2*D); i<imax; i++){

    int k = i - 2*D;

    if(factor[k]<2){
      temp=0;
      for(int j=0; j<D; j++){
        if(factor[j]>0)
          temp = temp + x[2*D+j]*pow(B[j*D+k],theta[j])*Omega_re[j*D+k]*pow(x[D+k],(1-theta[j]))
                       *pow(x[j],-(1-theta[j]))*x[j]*pow(A[j],(theta[j]-1))/x[D+j];
      }
      f[i - i0] = x[2*D+k] - temp;
    }
  }

  // lambda(334)=1
  if(3*D-1<imax)
    f[3*D-1-i0] = x[3*D-1]-1;

  // lambda(332)
  if(3*D-3>=i0 && 3*D-3<imax) {
    temp = 0;
    for(int j=0; j<D; j++){
      if(factor[j]==0)
        temp = temp + init_lambda[j]*(1-phi_htm[j])*(1-x[2*D+j]/x[D+j]/init_lambda[j]);
    }
    f[3*D-3 - i0] = x[3*D-3]- temp*x[3*D-1];
  }


  // lambda(333)
  if(3*D-2>=i0 && 3*D-2<imax) {
    temp=0;
    temp2=0;
    temp3=0;
    for(int j=0; j<D; j++){
      if(factor[j]==0)
        temp = temp + x[2*D+j];
      if(factor2[j]==1){
        temp2 = 0;
          for(int m=0; m<D; m++){
            if(factor[m]>0)
              temp2 = temp2 + pow(B[j*D+m],theta[j])*Omega_re[j*D+m]
                          *pow(x[D+m],(1-theta[j]))*pow(x[j],-(1-theta[j]))
                          *x[j]*pow(A[j],(theta[j]-1))/x[D+j];
          }
          temp3 = temp3 + x[2*D+j]*(1-temp2);
       }
    }
    f[3*D-2 - i0] = x[3*D-2] - (temp + temp3 - x[3*D-3]);
  }

  // p(200-331)
  for(int i=std::max(i0, D); i<std::min(2*D, imax); i++){
    int k = i - D;
    if(keynes[k] == 0)
      f[D+k - i0] = x[D+k] - x[2*D+k]/(A[k]*init_lambda[k]);
    else
      //f[D+k - i0] = x[D+k] - (1 + freq_re[k]*(x[k]-1));
      f[D+k - i0] = x[D+k] - exp(freq_re[k]*log(x[k]));
  }

  // p(334) = 1
  if(2*D-1>=i0 && 2*D-1<imax)
    f[2*D-1 - i0] = x[2*D-1] - 1;

  //-------------------------

  for(int i=0; i<imax-i0; i++)
    if(!std::isfinite(f[i])) {
      fprintf(stderr,"[%d] Oh no. i0 = %d, imax = %d, f[%d] = %e!\n", mpi_rank, i0, imax, i, f[i]);
      exit(-1);
    } 


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

  PetscPrintf(PETSC_COMM_WORLD, "- Computing the Jacobian matrix...\n");

  ProblemData *data = (ProblemData*)ctx;

//  PetscPrintf(PETSC_COMM_WORLD, "- Computing the Jacobian matrix...0...\n");

  data->UpdateXGlob(X);

//  PetscPrintf(PETSC_COMM_WORLD, "- Computing the Jacobian matrix...0.5...\n");

  int nVar = data->nVar;

  const double *f0, *f1;

  double *x = data->xglob;
  double dx = 1.0e-5;

  int i0, imax, my_size;
  VecGetOwnershipRange(X, &i0, &imax);
  my_size = imax - i0;

  for(int j=0; j<nVar; j++) {

    x[j] -= dx;
    data->xglob_ready = true;

//    PetscPrintf(PETSC_COMM_WORLD, "- Computing the Jacobian matrix...1...\n");

    Fun(tao, X, data->F0, ctx);

//    PetscPrintf(PETSC_COMM_WORLD, "- Computing the Jacobian matrix...2...\n");

    x[j] += 2.0*dx; 
    data->xglob_ready = true;

//    PetscPrintf(PETSC_COMM_WORLD, "- Computing the Jacobian matrix...3...\n");

    Fun(tao, X, data->F1, ctx);

 //   PetscPrintf(PETSC_COMM_WORLD, "- Computing the Jacobian matrix...4...\n");

    x[j] -= dx; //restore
 
    VecGetArrayRead(data->F0,&f0);
    VecGetArrayRead(data->F1,&f1);

    for(int i=0; i<my_size; i++)
      data->J[i*nVar+j] = (f1[i] - f0[i])/(2.0*dx); 

    VecRestoreArrayRead(data->F0,&f0);
    VecRestoreArrayRead(data->F1,&f1);
  }


  MatSetValues(J, my_size, data->idm, nVar, data->idn, (double*)data->J, INSERT_VALUES);

  MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);

  PetscPrintf(PETSC_COMM_WORLD, "- Done with Jacobian matrix...\n");

/*
  VecView(X,PETSC_VIEWER_STDOUT_WORLD);
  MatView(J,PETSC_VIEWER_STDOUT_WORLD);
  exit(-1);
*/

  return 0;
}
 
