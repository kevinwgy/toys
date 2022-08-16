#include <petscsnes.h>
#include "function.h"
#include <iostream>
#include <fstream>
using namespace std;

int main(int argc,char **argv)
{
  SNES           snes;         /* nonlinear solver context */
  KSP            ksp;          /* linear solver context */
  PC             pc;           /* preconditioner context */
  Vec            x,r;          /* solution, residual vectors */
  Mat            J;            /* Jacobian matrix */
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,NULL,NULL);


  ProblemData data; /* problem-specific data */
  SetProblemData(data);
  int nVar = GetNumberOfVariables(data);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  SNESCreate(PETSC_COMM_WORLD,&snes);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix and vector data structures; set corresponding routines
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  VecCreate(PETSC_COMM_WORLD,&x);
  VecSetSizes(x,PETSC_DECIDE,nVar);
  VecSetFromOptions(x);
  VecDuplicate(x,&r);

  MatCreate(PETSC_COMM_WORLD,&J);
  MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,nVar,nVar);
  MatSetFromOptions(J);
  MatSetUp(J);

  // Set function evaluation routine and vector. (and check for errors)
  // r = Fun(x)
  ierr = SNESSetFunction(snes,r,Fun,&data);  CHKERRQ(ierr);

  // Set Jacobian matrix data structure and Jacobian evaluation routine (and check for error)
  ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefault,NULL);  CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Set linear solver defaults for this problem. By extracting the
     KSP and PC contexts from the SNES context, we can then
     directly call any KSP and PC routines to set various options.
  */
  SNESGetKSP(snes,&ksp);
  KSPGetPC(ksp,&pc);
  PCSetType(pc,PCNONE);
  ierr = KSPSetTolerances(ksp,1.e-6,PETSC_DEFAULT,PETSC_DEFAULT,50);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  SetInitialGuess(x, data);

  // evaluation function using initial guess
  
  Vec tmp;
  VecDuplicate(x,&tmp);
  Fun(snes, x, tmp, &data);
  VecView(tmp,PETSC_VIEWER_STDOUT_WORLD);
  double mynorm(-1);
  VecNorm(tmp, NORM_2, &mynorm);
  fprintf(stderr,"Norm of f(x0) = %e.\n", mynorm);

  // Set options
  /*
     Set SNES/KSP/KSP/PC runtime options, e.g.,
         -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
     These options will override those specified above as long as
     SNESSetFromOptions() is called _after_ any other customization
     routines.
  */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);


  // ----------------------
  //  SOLVE
  // ----------------------
  cout << "- Solving the equation." << endl;
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  cout << "- Done!" << endl << endl;

/*
  cout << "Solution..." << endl;
  Vec f;
  VecView(x,PETSC_VIEWER_STDOUT_WORLD);
//  SNESGetFunction(snes,&f,0,0);
//  VecView(r,PETSC_VIEWER_STDOUT_WORLD);
*/
  cout << "F(X)..." << endl;
  Fun(snes, x, tmp, &data);
 // VecView(tmp,PETSC_VIEWER_STDOUT_WORLD);
  VecNorm(tmp, NORM_2, &mynorm);
  fprintf(stderr,"Norm of f(x) = %e.\n", mynorm);
   

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

  ierr = VecDestroy(&x);CHKERRQ(ierr); ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr); ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
