#pragma once
#include<cstdio>
#include<cstdlib>
#include<vector>
using namespace std;

int ReadVectorFromFile(const char *filename, vector<double> &v)
{
  FILE *file = fopen(filename, "r");
  if(file == NULL) {
    fprintf(stderr,"*** Error: Unable to open file %s.\n", filename);
    exit(-1);
  }

  v.clear();
  double s;
  int nread;
  while(v.size()<1.0e8) {
    nread = fscanf(file, "%lf", &s);
    if(nread!=1)
      break;
    v.push_back(s);
  }

  fprintf(stdout,"- Read a vector of length %d from %s. First: %e. Last: %e.\n", v.size(), filename, v.front(), v.back());
  
  fclose(file);

  return v.size();
}
