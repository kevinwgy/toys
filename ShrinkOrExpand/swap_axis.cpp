//-----------------------------------------------------------
//  Discription: exchange x-coordinates and y-coordinates 
//       Inputs: path to mesh file
//      Outputs: new mesh file -- output.top 
//       Author: Kevin Wang (created on Sep.16,2009)
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
  if(argc!=2){
    fprintf(stderr,"Format: [binary] <path to mesh file> \n");
    exit(-1);
  }
  ifstream inFile(argv[1],ios::in);
  ofstream outFile("output.top",ios::out); //app = append, out = output
  inFile.precision(6);
  outFile.precision(6);

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
    swap(x,y);
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
