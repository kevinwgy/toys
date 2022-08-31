#include <petsctao.h>
#include "function_test.h"
#include <iostream>
#include <fstream>
using namespace std;

int main(int argc,char **argv)
{
  Tao            tao;
  Vec            x, xl, xu, r;
  Mat            J;

  PetscInitialize(&argc,&argv,NULL,NULL);

  ProblemData data; /* problem-specific data */
  SetProblemData(data);
  int nVar = GetNumberOfVariables(data);

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

  fprintf(stderr,"2 nVar = %d.\n", nVar);
  SetInitialGuess(x, data);
  fprintf(stderr,"3 nVar = %d.\n", nVar);
  SetSolutionBounds(xl, xu, data);


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
  cout << "- Solving the optimization problem." << endl;
  TaoSolve(tao);
  cout << "- Done!" << endl << endl;


  TaoGetSolutionVector(tao, &x);

  // print to file
  const double *xx;
  VecGetArrayRead(x,&xx);
  ofstream out("solution.txt"); 
  if(out.is_open()) {

   for(int i=0; i<nVar; i++)
     out << xx[i] << "\n"; 

   out.close();
   
  } else
    cout << "*** Error: Unable to write solution to file 'solution.txt'." << endl;
  VecRestoreArrayRead(x,&xx);
 
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
