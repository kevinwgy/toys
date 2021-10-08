//-----------------------------------------------------------
//  Discription: This routine computes the 1,2,inf norm of difference.
//       Inputs: <matcher file> <solution 1> <solution 2>
//      Outputs: 1,2, and infinity norms of difference
//       Author: Kevin Wang (Jun.1,2010)
//        Notes: (WARNING) this code doesn't verify inputs.
//                         Only works for scalar solutions
//-----------------------------------------------------------
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <list>
using namespace std;

int main(int argc, char* argv[])
{
  //parsing
  if(argc!=4 && argc!=5) {
    cout<<"Format: [binary] <matcher file> <solution 1> <solution 2> [optional <cv 1>]" << endl;
    exit(-1);
  }
  ifstream matched(argv[1],ios::in);
  ifstream sol1(argv[2],ios::in);
  ifstream sol2(argv[3],ios::in);
  ifstream cv1;
  bool cvWeighted = (argc==5) ? true : false;
  if(cvWeighted)
    cv1.open(argv[4],ios::in);
    
  //global variables
  double norm1, norm2, norminf;
  int nSol1, nSol2, nMatched;

  //load solution 1
  double s;
  list<double> solution;
  list<double>::iterator it;
  
  while(1) {
    sol1 >> s;
    if(!sol1.eof())
      solution.push_back(s);
    else
      break;
  }
  nSol1 = solution.size();
  double S1[nSol1];
  int count = 0;
  for(it=solution.begin(); it!=solution.end(); it++) {
    S1[count] = *it;
    count++;
  }
  solution.clear();
  sol1.close();
  cout << nSol1 <<" data-points in solution 1 loaded." << endl;

  //load solution 2
  while(1) {
    sol2 >> s;
    if(!sol2.eof())
      solution.push_back(s);
    else
      break;
  }
  nSol2 = solution.size();
  double S2[nSol2];
  count = 0;
  for(it=solution.begin(); it!=solution.end(); it++) {
    S2[count] = *it;
    count++;
  }
  solution.clear();
  sol2.close();
  cout << nSol2 <<" data-points in solution 2 loaded." << endl;

  //load cv 1 (optional)
  double *CV;
  if(cvWeighted) {
    double tot = 0.0;
    CV = new double[nSol1];
    for(int i=0; i<nSol1; i++) {
      cv1 >> CV[i];
      tot += CV[i];
    }
    cv1.close();
    cout << nSol1 <<" data-points in cv 1 loaded. Total Volume = " << scientific << tot << endl;
  }
  else
    CV = 0;
 
  //compute norms of S1-S2
  double x,y,z,delta,area;
  char word1[30];
  int i1, i2, maxId = 0;
  norm1 = norm2 = norminf = 0.0;
  count = 0;
  while(!matched.eof()) {
    word1[0] = 0;
    matched >> word1;
    i1 = strtol(word1,NULL,10);
    if(i1==0) break;
    matched >> i2 >> x >> y >> z;
    i1--;
    i2--;

    if(x>-2 && x<3 && z>-2 && z<2)
      continue;

    count++;
    delta = S1[i1]-S2[i2];
    area = cvWeighted ? CV[i1] : 1.0;

    norm1 += area*fabs(delta);
    norm2 += area*delta*delta;
    if(fabs(delta)>norminf) {
      norminf = fabs(delta);
      maxId = i1+1;
    }
  }
  norm2 = sqrt(norm2);
  matched.close();
  fprintf(stderr,"|S1-S2| with dim = %d computed.\n", count);
  if(cvWeighted)
    fprintf(stderr,"-- control volume taken into account.\n");
  fprintf(stderr,"  1-norm: %e\n", norm1);
  fprintf(stderr,"  2-norm: %e\n", norm2);
  if(!cvWeighted)
    fprintf(stderr,"inf-norm: %e (at Node %d in Solution 1)\n", norminf, maxId);

  // clean-up
  if(CV) delete[] CV;
  return 0;
}
