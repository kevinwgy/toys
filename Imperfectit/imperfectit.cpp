//-----------------------------------------------------------
//  Discription: geometiric imperfection in cylinder
//       Inputs: <path to mesh file> <size of imperfection (%)> <mode>
//      Outputs: new mesh file -- output.top
//       Author: Kevin Wang (created on Sep.30,2009)
//        Notes: (WARNING) this code doesn't verify inputs.
//               (WARNING) x-direction should be aligned with cylinder axis 
//-----------------------------------------------------------

#include<stdio.h>
#include<math.h>
#include<iostream>
#include<fstream>
using namespace std;

int main(int argc, char* argv[])
{
  if(argc!=4) {
    fprintf(stderr,"format: [binary] [path to mesh file] [size of imperfection (%)] [mode]\n");
    exit(-1);
  }

  double impSize = 1.0/100.0*atof(argv[2]); //e.g. 0.01;
  int impMode = atoi(argv[3]);    //e.g. 4;
  
  ifstream inFile(argv[1],ios::in);
  ofstream outFile("output.top",ios::out); //app = append, out = output
  inFile.precision(6);
  outFile.precision(6);
 
  double x, y, z;
  int index;
  double theta;
  double alpha;
  char word1[20], word2[20];

  inFile >> word1 >> word2;
  outFile << word1 << " " << word2 << endl;

  while(!inFile.eof()) {
    inFile >> word1;
    index = strtol(word1,NULL,10);
    if(index==0) {outFile << word1; break;}
    inFile >> x >> y >> z;

    theta = atan2(z,y);
    alpha = 1.0 - impSize*cos(impMode*theta);
    y = alpha*y;
    z = alpha*z;

    outFile << index << " " << scientific << x << " " << y << " " << z << endl;
  }
  fprintf(stderr,"DONE with nodes.\n");

  //read the rest of file and write to output without any modification.
  string line;
  while(getline(inFile, line))
    outFile << line << endl;

  inFile.close();
  outFile.close();
 
  return 0;
}
