#pragma once

#include<petscsnes.h>
#include<cmath>

//-------------------------------------
// Return the number of variables
//-------------------------------------
int GetNumberOfVariables()
{
  return 2;
}

//-------------------------------------
// Set initial guess
//-------------------------------------
void SetInitialGuess(Vec X)
{
  double *x;
  VecGetArray(X,&x);

  // Set initial guess
  x[0] = 0.0;
  x[1] = 0.0;

  VecRestoreArray(X,&x);
}

//-------------------------------------
// Evaluation function F = Fun(X)
//-------------------------------------
PetscErrorCode Fun(SNES snes,Vec X,Vec F,void *dummy)
{
  const double *x;
  double *f;
  VecGetArrayRead(X,&x);
  VecGetArray(F,&f);


 //-----------------------------
 // Compute function
 
 f[0] = exp(-exp(-x[0]-x[1])) - x[1]*(1.0+x[0]*x[0]);
 f[1] = x[0]*cos(x[1]) + x[1]*sin(x[0]) - 0.5;
  
 //----------------------------

  VecRestoreArrayRead(X,&x);
  VecRestoreArray(F,&f);
  return 0;
}
