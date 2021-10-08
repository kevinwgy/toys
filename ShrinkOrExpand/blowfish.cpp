//-----------------------------------------------------------
//  Discription: blowfish can displace, expand, or shrink...
//               shrink or expand a mesh w.r.t. one point
//       Inputs: <path to mesh file> 
//               <center-x> <center-y> <center-z> 
//               <coeff-x>  <coeff-y>  <coeff-z> 
//               <dx>       <dy>       <dz>
//      Outputs: new mesh file -- output.top 
//       Author: Kevin Wang (created on Sep.16,2009)
//        Notes: (WARNING) this code doesn't verify inputs.
//               (WARNING) this code assumes the mesh file
//                         starts with the node set
//-----------------------------------------------------------

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
using namespace std;

int main (int argc, char* argv[]) 
{
  //global variables & parsing
  if((argc!=8)&&(argc!=11)){
    fprintf(stderr,"Format: [binary] <path to mesh file> < center-x> <center-y> <center-z>\n");
    fprintf(stderr,"                 <coeff-x> <coeff-y> <coeff-z> \n");
    fprintf(stderr,"                 (optional) <dx> <dy> <dz>\n");
    exit(-1);
  }
  ifstream inFile(argv[1],ios::in);
  ofstream outFile("output.top",ios::out); //app = append, out = output
  inFile.precision(6);
  outFile.precision(6);
  double Ox = 0.0, Oy = 0.0, Oz = 0.0;
  double cx = 1.0, cy = 1.0, cz = 1.0;
  double dx = 0.0, dy = 0.0, dz = 0.0;
  Ox = atof(argv[2]); 
  Oy = atof(argv[3]);
  Oz = atof(argv[4]);
  cx = atof(argv[5]);
  cy = atof(argv[6]);
  cz = atof(argv[7]);
  if(argc==11) {
    dx = atof(argv[8]);
    dy = atof(argv[9]);
    dz = atof(argv[10]);
  }

  char word1[20], word2[20];
  int integer1;
  double x,y,z;

  // load, process and output the node set
  int nNodes;
  inFile >> word1 >> word2;
  outFile << word1 << " " << word2 << endl;

  while(!inFile.eof()) {
    inFile >> word1;
    integer1 = strtol(word1,NULL,10);
    if(integer1==0) {outFile << word1; break;}
    inFile >> x >> y >> z;
    x = cx*(x - Ox) + Ox + dx;
    y = cy*(y - Oy) + Oy + dy;
    z = cz*(z - Oz) + Oz + dz;
    outFile.precision(8);
    outFile << integer1 << " " << scientific << x << " " << y << " " << z << endl; 
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
