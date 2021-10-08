#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
using namespace std;

int main (int argc, char* argv[]) 
{
  //global variables & parsing
  if(argc!=3 && argc!=5 && argc!=6 && argc!=7){
    fprintf(stderr,"Format: [binary] <path to solution file> <path to output file prefix> {(optional) <start time> <end time> <lowerbound> <exponent>}\n");
    exit(-1);
  }
  ifstream solFile(argv[1],ios::in);
  char fname1[256], fname2[256], fname3[256];
  sprintf(fname1, "%s_max.dat",argv[2]);
  sprintf(fname2, "%s_avg.dat",argv[2]);
  sprintf(fname3, "%s_int.dat",argv[2]);
  ofstream outFile1(fname1,ios::out);
  ofstream outFile2(fname2,ios::out);
  ofstream outFile3(fname3,ios::out);
  outFile1.precision(6);
  outFile2.precision(6);
  outFile3.precision(6);
  double tstart = -1e16;
  double tend = 1e16;
  if(argc==5 || argc==6 || argc==7) {
    tstart = atof(argv[3]);
    tend   = atof(argv[4]);
  }
  double sol_min = -1e16; //For example, this could be the threshold stress in the Tuler-Butcher failure criterion.
  if(argc==6 || argc==7) 
    sol_min = atof(argv[5]);
  double exp = 1;
  if(argc==7)
    exp = atof(argv[6]);


  //read the rest of file and write to output without any modification.
  string line;
  char word1[256], word2[256], word3[256], word4[256], word5[256], word6[256];
  int nNodes;
  solFile >> word1 >> word2 >> word3 >> word4 >> word5 >> word6;
  outFile1 << word1 << " " << word2 << "_max" << " " << word3 << " " << word4 << " " << word5 << " " << word6 << endl;
  outFile2 << word1 << " " << word2 << "_avg" << " " << word3 << " " << word4 << " " << word5 << " " << word6 << endl;
  outFile3 << word1 << " " << word2 << "_int" << " " << word3 << " " << word4 << " " << word5 << " " << word6 << endl;
  solFile >> nNodes; 
  outFile1 << nNodes << endl;
  outFile2 << nNodes << endl;
  outFile3 << nNodes << endl;
  outFile1 << "0.0" << endl; //time (useless)
  outFile2 << "0.0" << endl; //time (useless)
  outFile3 << "0.0" << endl; //time (useless)
  getline(solFile,line); // get the EOL
  fprintf(stderr,"nNodes = %d.\n", nNodes);
  double *Max = new double[nNodes];
  double *Avg = new double[nNodes];
  double *Int = new double[nNodes];
  double *Prev = new double[nNodes]; //for integration
  for(int i=0; i<nNodes; i++) {
    Max[i] = -1.0e16;
    Avg[i] = 0.0;
    Int[i] = 0.0;
    Prev[i] = 0.0;
  }

  double t0 = 0;
  double dt = 0, tprev = 0, tnow = 0;
  double v;
  int iter = 0;
  bool called = false;
  while(!solFile.eof()) {
    double tnow0;
    tprev = tnow;
    solFile >> tnow0;
    if(!solFile) break;
    if(tnow0 >= tend + 1.0e-15) break;
    tnow = tnow0;
    if(!called && tnow0 >= tstart - 1.0e-15) {
      t0 = tnow;
      called = true;
    }
    for(int i=0; i<nNodes; i++) {
      solFile >> v;  //THE VALUE
      if(!called)
        continue;
      if(v>Max[i])  Max[i] = v;
      if(iter && tprev >= tstart - 1.0e-15) {//integrate
        Avg[i] += (tnow - tprev)*0.5*(Prev[i] + v);
        if(v>sol_min) {
          if(argc>=6)
            Int[i] += (tnow - tprev)*pow(0.5*(Prev[i] + v) - sol_min, exp);
          else
            Int[i] += (tnow - tprev)*0.5*(Prev[i] + v);
//        cout << "tnow = " << tnow << ", tprev = " << tprev << endl;
        }
      }
      Prev[i] = v;
    } 
    iter++;
    getline(solFile,line); // get the EOL
    if(!(tnow >= tstart - 1.0e-15))
      cout << "Skipped it. " << iter << ": t = " << tnow << endl;
    else 
      cout << "Done with it. " << iter << ": t = " << tnow << endl;
  }


  // output
  for(int i=0; i<nNodes; i++) {
    outFile1 << Max[i] << endl;
    outFile2 << 1.0/(tnow-t0)*Avg[i] << endl;
    outFile3 << Int[i] << endl;
  }

  cout << "Start time: " << t0 << ", Final Time: " << tnow << endl;
  fprintf(stderr,"DONE.\n");

  solFile.close();
  outFile1.close();
  outFile2.close();
  outFile3.close();

  delete[] Max;
  delete[] Avg;
  delete[] Int;
  delete[] Prev;
  return 0;
}
