#ifndef _OUTPUT_H_
#define _OUTPUT_H_
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
using namespace std;
struct Input;

class Output {
  char filename_base[128];
  char full_filename_base[128];
  ofstream summaryfile;
  ofstream solfile;

public:
  Output(Input *input);
  ~Output();

  void output_solution(vector<double> &alpha, vector<double> &intensity, 
                       double xmin, double xmax, int N);
  const string getCurrentDateTime();
};
#endif
