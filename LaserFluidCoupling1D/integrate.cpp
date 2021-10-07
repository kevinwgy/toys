#include <iostream>
#include <vector>
#include <math.h>
#include "input.h"
using namespace std;

void ForwardEuler(Input &input, double dx, vector<double> &alpha, vector<double> &intensity)
{
  for(int i=1; i<alpha.size(); i++)
    intensity[i] = intensity[i-1] - dx*(alpha[i-1]*intensity[i-1]);
  return;
}

void RungeKutta2(Input &input, double dx, vector<double> &alpha, vector<double> &intensity)
{
  //TODO: Xuning
  cout << "Not implemented yet!" << endl;
  return;
}

void integrate(Input &input, double dx, vector<double> &alpha, vector<double> &intensity)
{
  ForwardEuler(input, dx, alpha, intensity);
  //TODO: Xuning: After implementing 2nd-order Runge-Kutta (or another integrator), should use it here.
  //RungeKutta2(input, dx, alpha, intensity);

  return;
}
