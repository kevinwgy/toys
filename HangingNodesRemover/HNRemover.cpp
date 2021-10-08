//-----------------------------------------------------------
//  Discription: This routine removes hanging nodes from a mesh
//       Inputs: <path to mesh file> <# nodes> <# elements>
//      Outputs: New mesh file -- output.top
//       Author: Kevin Wang (created on Feb.21,2009)
//        Notes: (WARNING) This code doesn't verify inputs.
//-----------------------------------------------------------

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>
using namespace std;

struct Vec3D {
  double x,y,z;
  Vec3D() {x = y = z = 0.0;}
  Vec3D(double _x, double _y, double _z) {x = _x; y = _y; z = _z;}
  ~Vec3D() {}
  Vec3D &operator=(const Vec3D &v) {x = v.x; y = v.y; z = v.z;}
};

int main(int argc, char* argv[])
{
  if(argc!=4) {
    cout << "Incorrect usage!" << endl;
    cout << "<binary> <path to mesh file> <# nodes> <# elements>" << endl;
    exit(-1);
  }

  map<int,int> eMap;
  eMap[1]   = 2;
  eMap[4]   = 3;
  eMap[5]   = 4;
  eMap[106] = 2;
  eMap[104] = 3;
  eMap[108] = 3;
  eMap[188] = 4;
  const int eMax = 4;
  map<int,int>::iterator it;
  map<int,Vec3D> X;

  ifstream inFile(argv[1],ios::in);
  ofstream outFile("output.top",ios::out);
  outFile.precision(12);

  int nNodes = atoi(argv[2]);
  int nElems = atoi(argv[3]);
  int **E = new int*[nElems];
  for(int i=0; i<nElems; i++)
    E[i] = new int[eMax];
  int *code = new int[nElems];

  char word1[20], word2[20], word3[20], word4[20];
  int i1, i2, i3, i4;

  // load inputfile 
  inFile >> word1 >> word2;
  outFile << word1 << " " << word2 << endl;

  double x, y, z;
  for(int i=0; i<nNodes; i++) {
    inFile >> i1 >> x >> y >> z;
    X[i1] = Vec3D(x,y,z);
  }

  inFile >> word1 >> word2 >> word3 >> word4;

  cout << word1 << " " << word2 << " " << word3 << " " << word4 << endl;

  for(int i=0; i<nElems; i++) {

    inFile >> i1 >> i2;
    it = eMap.find(i2);
    if(it==eMap.end()) {
      cout << "ERROR: Doesn't understand element type " << i2 << "." << endl;
      exit(-1);
    } else
      code[i] = i2;
 
    int nn = eMap[code[i]];
    for(int j=0; j<nn; j++)
      inFile >> E[i][j];
  }


  // process...
  map<int,int> new2old;
  map<int,int> old2new;
  int count = 0;

  for(int i=0; i<nElems; i++) {
    int nn = eMap[code[i]];
    for(int j=0; j<nn; j++) {
      it = old2new.find(E[i][j]);
      if(it==old2new.end()) {
        count++;
        old2new[E[i][j]] = count;
        new2old[count]   = E[i][j];
      }
    }
  }

  cout << count << " nodes in total." << endl;

  // output
  for(int i=0; i<count; i++) {
    int myNode = new2old[i+1];
    outFile << i+1 << scientific << " " << X[myNode].x << " " << X[myNode].y << " " << X[myNode].z << endl;
  }

  outFile << word1 << " " << word2 << " " << word3 << " " << word4 << endl;
  for(int i=0; i<nElems; i++) {
    outFile << i+1 << " " << code[i];
    int nn = eMap[code[i]];
    for(int j=0; j<nn; j++)
      outFile << " " << old2new[E[i][j]];
    outFile << endl;
  }

  inFile.close();
  outFile.close();
  return 0;
}
