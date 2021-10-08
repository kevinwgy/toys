//-----------------------------------------------------------
//  Discription: This routine matches two meshes.
//       Inputs: <path to mesh 1> <path to mesh 2> <tolerance>
//      Outputs: <matched> <nonmatched> (based on mesh 1)
//       Author: Kevin Wang (Jun.1,2010)
//        Notes: (WARNING) this code doesn't verify inputs.
//-----------------------------------------------------------
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <list>
#include "KDTree.h"
using namespace std;

struct Vec3D {
  double x,y,z;
};

class MyNode {
  int id;
  double x[3], w[3];
public:
  MyNode() {}
  MyNode(int i, double _x, double _y, double _z, double _w = 0.0) {
    id = i;
    x[0] = _x;
    x[1] = _y;
    x[2] = _z;
    w[0] = w[1] = w[2] = _w;
  }
  double val(int i) const { return x[i]; }
  double width(int i) const { return w[i]; }
  int trId() const { return id; }
};

int main(int argc, char* argv[])
{
  //parsing
  if(argc!=4) { 
    cout<<"Format: [binary] <mesh 1> <mesh 2> <tolerance>" << endl;
    exit(-1);
  }
  ifstream mesh1(argv[1],ios::in);
  ifstream mesh2(argv[2],ios::in);
  ofstream matched("Matched.top",ios::out);
  ofstream nonmatched("NonMatched.top",ios::out);
  double TOL = atof(argv[3]);

  //global variables
  int nNodes1, nNodes2;

  //load mesh 1
  char word1[50], word2[50], word3[50];
  int i1,i2;
  list<pair<int,Vec3D> > nodes;
  list<pair<int,Vec3D> >::iterator it;
  double x,y,z;

  mesh1 >> word1 >> word2;
  nNodes1 = 0;
  while(!mesh1.eof()) {
    mesh1 >> word1;
    i1 = strtol(word1,NULL,10);
    if(i1==0) break;
    Vec3D p;
    mesh1 >> p.x >> p.y >> p.z;
    nodes.push_back(pair<int,Vec3D>(i1,p));
    nNodes1++;
  }
  double X1[nNodes1][3];
  for(it=nodes.begin(); it!=nodes.end(); it++) {
    X1[it->first-1][0] = (it->second).x;
    X1[it->first-1][1] = (it->second).y;
    X1[it->first-1][2] = (it->second).z;
  }
  nodes.clear();
  mesh1.close();
  cout << nNodes1 <<" nodes in Mesh #1 loaded." << endl;

  //load mesh2
  mesh2 >> word1 >> word2;
  nNodes2 = 0;
  while(!mesh2.eof()) {
    mesh2 >> word1;
    i2 = strtol(word1,NULL,10);
    if(i2==0) break;
    Vec3D p;
    mesh2 >> p.x >> p.y >> p.z;
    nodes.push_back(pair<int,Vec3D>(i2,p));
    nNodes2++;
  }
  double X2[nNodes2][3];
  for(it=nodes.begin(); it!=nodes.end(); it++) {
    X2[it->first-1][0] = (it->second).x;
    X2[it->first-1][1] = (it->second).y;
    X2[it->first-1][2] = (it->second).z;
  }
  nodes.clear();
  mesh2.close();
  cout << nNodes2 <<" nodes in Mesh #2 loaded." << endl;

  //store mesh 2 in KDTree
  MyNode myNodes[nNodes2];
  for(int i=0; i<nNodes2; i++)
    myNodes[i] = MyNode(i, X2[i][0], X2[i][1], X2[i][2], 0.0);
  KDTree<MyNode> Michel(nNodes2,myNodes);

  //match mesh 2 with mesh 1 
  int maxCandy = 50;
  MyNode candy[maxCandy];

  bool found = false;
  double dist;
  int ID;
  int infofreq = nNodes1/100;
  int accum = 0;
  for(int i=0; i<nNodes1; i++) {
    found = false;
    dist = max(1.0e8*TOL,10.0);

    int nCand = Michel.findCandidatesWithin(X1[i], candy, maxCandy, TOL);
    if(nCand>maxCandy) {
      fprintf(stderr,"WARNING: For Node %d in Mesh 1, %d candidates found within dist = %e\n", i+1, nCand, TOL);
      nCand = maxCandy;
    }

    for(int j=0; j<nCand; j++) {
      int jCand = candy[j].trId();
      double dist1 = sqrt((X1[i][0]-X2[jCand][0])*(X1[i][0]-X2[jCand][0])
                         +(X1[i][1]-X2[jCand][1])*(X1[i][1]-X2[jCand][1])
                         +(X1[i][2]-X2[jCand][2])*(X1[i][2]-X2[jCand][2]));
      if(dist1<dist) {
        dist = dist1;
        found = true;
        ID = jCand;
        break;
      }
    }

    if(found) 
      matched << i+1 << " " << ID+1 << " " << X1[i][0] << " " << X1[i][1] << " " << X1[i][2] << endl;
    else {
      nonmatched << i+1 << " " << X1[i][0] << " " << X1[i][1] << " " << X1[i][2] << endl;
      fprintf(stderr,"Node %d in Mesh 1 (%e %e %e) cannot be matched from Mesh 2.\n", i+1, X1[i][0], X1[i][1], X1[i][2]);
    }

    if(accum++==infofreq) {
      accum =0;
      fprintf(stderr,"Finshed %d out of %d nodes.\r", i+1, nNodes1);
    }
  } 
  fprintf(stderr,"\n");
  fprintf(stderr,"Done\n");

  return 0;
}
