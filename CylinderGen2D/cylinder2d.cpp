#include<iostream>
#include<fstream>
#include<math.h>
using namespace std;

int main(int argc, char* argv[])
{
  // User's inputs
  // Unit: mm
  double R = 0.5*38.2;
  double wallthickness = 0.711;
  int nNodes_oneLayer = 160;
  double xmin = -0.15;
  double xmax = 0.15;

  double imperfect = 0.5/100.0; //0.5/100 --> 0.5%
  int impMode = 2;

  // Initialization
  ofstream meshfile("mesh.include",ios::out); 

  // create nodes.
  int nNodes = 2*nNodes_oneLayer;
  double nodes[nNodes][3];

  double pi = acos(0)*2.0;
  double da = 2.0*pi/nNodes_oneLayer;
  double theta, alpha;
  for (int i=0; i<nNodes_oneLayer; i++) {
    nodes[i][0] = xmin;
    nodes[i][1] = R*cos(i*da);
    nodes[i][2] = R*sin(i*da);
    // apply imperfection
    theta = atan2(nodes[i][2],nodes[i][1]);
    alpha = 1.0 - imperfect*cos(impMode*theta);
    nodes[i][1] *= alpha;
    nodes[i][2] *= alpha;
    // create the second layer of nodes
    nodes[i+nNodes_oneLayer][0] = xmax;
    nodes[i+nNodes_oneLayer][1] = nodes[i][1];
    nodes[i+nNodes_oneLayer][2] = nodes[i][2];
  }
  meshfile << "NODES" << endl;
  for(int i=0; i<nNodes; i++) {
    meshfile.width(8);
    meshfile << i+1;
    meshfile.width(18);
    meshfile.precision(8);
    meshfile << scientific << nodes[i][0];
    meshfile.width(18);
    meshfile.precision(8);
    meshfile << scientific << nodes[i][1];
    meshfile.width(18);
    meshfile.precision(8);
    meshfile << scientific << nodes[i][2] << "\n";
  }
  meshfile << "*" << endl;

  // Create elements
  meshfile << "TOPOLOGY" << endl;
  for(int i=0; i<nNodes_oneLayer; i++) {
    meshfile.width(8);
    meshfile << i+1;
    meshfile << "  16";
    meshfile.width(8);
    meshfile << i+1;
    meshfile.width(8);
    meshfile << i+1 + nNodes_oneLayer;
    meshfile.width(8);
    int next = (i+2>nNodes_oneLayer) ? 1 : i+2;
    meshfile << next + nNodes_oneLayer;
    meshfile.width(8);
    meshfile << next << "\n";
  }
  meshfile << "*" << endl;

  // Create surfacetopo for embedded surface
  meshfile << "SURFACETOPO 8" << endl;
  int counter = 1;
  for(int i=0; i<nNodes_oneLayer; i++) {
    meshfile.width(8);
    meshfile << counter;
    meshfile << "  3";
    meshfile.width(8);
    meshfile << i+1;
    meshfile.width(8);
    meshfile << i+1 + nNodes_oneLayer;
    meshfile.width(8);
    int next = (i+2>nNodes_oneLayer) ? 1 : i+2;
    meshfile << next + nNodes_oneLayer << "\n";
    counter++;
    meshfile.width(8);
    meshfile << counter;
    meshfile << "  3";
    meshfile.width(8);
    meshfile << i+1;
    meshfile.width(8);
    next = (i+2>nNodes_oneLayer) ? 1 : i+2;
    meshfile << next + nNodes_oneLayer;
    meshfile.width(8);
    meshfile << next << "\n";
    counter++;
  }
  meshfile << "*" << endl;

  // write contact surface
  meshfile << "SURFACETOPO 1 SURFACE_THICKNESS " << wallthickness << endl;
  for(int i=0; i<nNodes_oneLayer; i++) {
    meshfile.width(8);
    meshfile << i+1;
    meshfile << "  1";
    meshfile.width(8);
    meshfile << i+1;
    meshfile.width(8);
    int next = (i+2>nNodes_oneLayer) ? 1 : i+2;
    meshfile << next;
    meshfile.width(8);
    meshfile << next + nNodes_oneLayer;
    meshfile.width(8);
    meshfile << i+1 + nNodes_oneLayer << "\n";
  }
  meshfile << "*" << endl;

  meshfile.close();
  return 0;
}
