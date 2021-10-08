#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
using namespace std;

int main (int argc, char* argv[]) 
{
  //global variables & parsing
  if(argc!=6){
    fprintf(stderr,"Format: [binary] <path to mesh file> <path to solution file> <x> <y> <z>\n");
    exit(-1);
  }
  ifstream inFile(argv[1],ios::in);
  ifstream solFile(argv[2],ios::in);
  ofstream outFile("nodal_probe.txt",ios::out); //app = append, out = output
  outFile.precision(6);
  double Ox = 0.0, Oy = 0.0, Oz = 0.0;
  Ox = atof(argv[3]); 
  Oy = atof(argv[4]);
  Oz = atof(argv[5]);

  char word1[20], word2[20];
  int nNodes, integer1;
  double x,y,z;

  // load, process and output the node set
  inFile >> word1 >> word2;

  int id;
  double minDist = 1.0E12;
  double dist = 0, Sx, Sy, Sz;
  while(!inFile.eof()) {
    inFile >> word1;
    integer1 = strtol(word1,NULL,10);
    if(integer1==0) break;
    inFile >> x >> y >> z;
    dist = sqrt((x-Ox)*(x-Ox) + (y-Oy)*(y-Oy) + (z-Oz)*(z-Oz));
    if(dist<minDist) {
      minDist = dist;
      Sx = x; Sy = y; Sz = z;
      id = integer1;
    }
  }
  fprintf(stderr,"Closest node: %d  %e  %e  %e; dist = %e.\n", id, Sx, Sy, Sz, minDist);
  outFile << "# Node " << id << ": " << Sx << " " << Sy << " " << Sz << endl;
  fprintf(stderr,"Extracting solution data.\n");

  //read the rest of file and write to output without any modification.
  string line;
  getline(solFile, line);
  solFile >> nNodes; 
  getline(solFile,line); // get the EOL
  fprintf(stderr,"nNodes = %d.\n", nNodes);
  while(!solFile.eof()) {
    getline(solFile, line); //time
    outFile << line << "  ";
    for(int i=0; i<nNodes; i++) {
      getline(solFile, line);
      if(i==id-1)
        outFile << line << endl;
    } 
  }
  fprintf(stderr,"DONE.\n");

  inFile.close();
  solFile.close();
  outFile.close();
  return 0;
}
