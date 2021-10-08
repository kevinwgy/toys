#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
using namespace std;

int main (int argc, char* argv[]) 
{
  char fname1[256], fname2[256], fname3[256];
  sprintf(fname1, "max.dat");
  ofstream outFile1(fname1,ios::out);
  FILE *myfile = fopen("max.dat","w");

  int nNodes = 1000000;

  // output
  for(int i=0; i<nNodes; i++) {
    outFile1 << 1.0 << endl;
//    fprintf(myfile,"%e\n", 1.0);
  }

  fprintf(stderr,"DONE.\n");

  outFile1.close();
  fclose(myfile);

  return 0;
}
