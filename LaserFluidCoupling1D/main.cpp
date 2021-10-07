/**********************************************************************************
 * Copyright ©Xuning Zhao, Kevin G. Wang, 2019
 * (1) Redistribution and use in source and binary forms, with or without modification,
 *     are permitted, provided that this copyright notice is retained.
 * (2) Use at your own risk.
 **********************************************************************************/
#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <omp.h>
#include "input.h"
#include "output.h"
#include "hgversion.h"
using namespace std;
int omp_get_thread_num();

void integrate(Input &input, double dx, vector<double> &alpha, vector<double> &intensity);
//--------------------------------------------------------------
// Main Function
//--------------------------------------------------------------
int main(int argc, char* argv[])
{
  clock_t start_time = clock(); //for timing purpose only

  cout << " ============================================== " << endl;
  cout << " RUNNING A ONE-DIMENSIONAL LASER-FLUID ANALYSIS " << endl;
  cout << " Revision: " << hgRevisionNo << " | " << hgRevisionHash <<endl;
  cout << " ============================================== " << endl;

  //--------------------------------------------------------------
  // Inputs
  //--------------------------------------------------------------
  Input input;
  input.readCmdLine(argc, argv);
  input.readCmdFile();

  //--------------------------------------------------------------
  // Allocate memory, initialization
  //--------------------------------------------------------------
  vector<double> alpha(input.file.N, input.file.alpha0);
  vector<double> intensity(input.file.N, 0.0);

  double dx = (input.file.xmax - input.file.xmin)/(input.file.N-1);
  for(int i=0; i<alpha.size(); i++) {
    double x = input.file.xmin + i*dx;
    if(x>=input.file.interval1_xmin && x<input.file.interval1_xmax)
        alpha[i] = input.file.interval1_alpha;
    if(x>=input.file.interval2_xmin && x<input.file.interval2_xmax)
        alpha[i] = input.file.interval2_alpha;
    if(x>=input.file.interval3_xmin && x<input.file.interval3_xmax)
        alpha[i] = input.file.interval3_alpha;
  }

  intensity[0] = input.file.I0; //source condition

  //--------------------------------------------------------------
  // Solve the equation
  //--------------------------------------------------------------
  integrate(input, dx, alpha, intensity);

  //--------------------------------------------------------------
  // Output
  //--------------------------------------------------------------
  Output output(&input);
  output.output_solution(alpha, intensity, input.file.xmin, input.file.xmax, input.file.N);
   

  cout << "Successful Completion." << endl;
  cout << "Total Computation Time: " << ((double)(clock()-start_time))/CLOCKS_PER_SEC << " sec." << endl;
  return 0;
}

