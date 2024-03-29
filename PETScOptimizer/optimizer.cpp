#include <petsctao.h>
#include "function.h"
#include <iostream>
#include <fstream>
using namespace std;

int main(int argc,char **argv)
{
  Tao            tao;
  Vec            x, xl, xu, r;
  Mat            J;

  PetscInitialize(&argc,&argv,NULL,NULL);

  int mpi_rank, mpi_size;
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);

  ProblemData data; /* problem-specific data */
  SetProblemData(data);
  int nVar = GetNumberOfVariables(data);

  if(!mpi_rank)
    fprintf(stderr,"nVar = %d.\n", nVar);

  TaoCreate(PETSC_COMM_WORLD,&tao);

  VecCreate(PETSC_COMM_WORLD,&x);
  VecCreate(PETSC_COMM_WORLD,&xl);
  VecCreate(PETSC_COMM_WORLD,&xu);
  VecCreate(PETSC_COMM_WORLD,&r);
  VecSetSizes(x,PETSC_DECIDE,nVar);
  VecSetSizes(xl,PETSC_DECIDE,nVar);
  VecSetSizes(xu,PETSC_DECIDE,nVar);
  VecSetSizes(r,PETSC_DECIDE,nVar);
  VecSetFromOptions(x);
  VecSetFromOptions(xl);
  VecSetFromOptions(xu);
  VecSetFromOptions(r);

  MatCreate(PETSC_COMM_WORLD,&J);
  MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,nVar,nVar);
  MatSetFromOptions(J);
  MatSetUp(J);

  SetInitialGuess(x, data);
  SetSolutionBounds(xl, xu, data);

  Fun(tao, x, r, (void*)&data);
  VecView(r, PETSC_VIEWER_STDOUT_WORLD); 
  double rnorm;
  VecNorm(r, NORM_2, &rnorm);
  if(!mpi_rank)
    cout << "- Initial guess: 2-Norm of residual is: " << rnorm << endl;
  VecNorm(r, NORM_INFINITY, &rnorm);
  if(!mpi_rank)
    cout << "- Initial guess: inf-Norm of residual is: " << rnorm << endl;



  // Set up solver
  TaoSetType(tao, TAOBRGN);
  TaoSetInitialVector(tao, x);
  TaoSetVariableBounds(tao, xl, xu);
  TaoSetResidualRoutine(tao, r, Fun, (void*)&data);
  TaoSetJacobianResidualRoutine(tao, J, J, Jacobian, (void*)&data);
  TaoSetTolerances(tao, 1.0e-6, 1.0e-6, 1.0e-6);
  TaoSetFromOptions(tao);

  // ----------------------
  //  SOLVE
  // ----------------------
  if(!mpi_rank)
    cout << "- Solving the optimization problem." << endl;

  TaoSolve(tao);

  if(!mpi_rank)
    cout << "- Done!" << endl << endl;


  TaoGetSolutionVector(tao, &x);

  // print to file
  PetscViewer viewer;
  PetscViewerASCIIOpen(PETSC_COMM_WORLD, "Solution.m", &viewer);
  PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
  VecView(x,viewer);
  PetscViewerPopFormat(viewer);
  PetscViewerDestroy(&viewer);


  Fun(tao, x, r, (void*)&data);
  VecView(r, PETSC_VIEWER_STDOUT_WORLD); 
  VecNorm(r, NORM_2, &rnorm);
  if(!mpi_rank)
    cout << "- 2-Norm of residual is: " << rnorm << endl;
  VecNorm(r, NORM_INFINITY, &rnorm);
  if(!mpi_rank)
    cout << "- inf-Norm of residual is: " << rnorm << endl;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  VecDestroy(&x);
  VecDestroy(&xl);
  VecDestroy(&xu);
  VecDestroy(&r);

  PetscFinalize();

  return 0;
}
