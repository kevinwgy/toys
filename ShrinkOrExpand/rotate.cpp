//-----------------------------------------------------------
//  Discription: rotate a mesh w.r.t. an axis (x,y, or z) 
//       Inputs: <path to mesh file> 
//               <axis (x, y, or z)> <degree>
//      Outputs: new mesh file -- output.top 
//       Author: Kevin Wang (created on Jan.27,2009)
//        Notes: (WARNING) this code doesn't verify inputs.
//               (WARNING) this code assumes the mesh file
//                         starts with the node set
//-----------------------------------------------------------

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#define PI 3.14159265
using namespace std;

int main (int argc, char* argv[]) 
{
  //global variables & parsing
  if(argc!=4){
    fprintf(stderr,"Format: [binary] <path to mesh file> <axis (x, y, or z)> <degree>\n");
    exit(-1);
  }
  ifstream inFile(argv[1],ios::in);
  ofstream outFile("output.top",ios::out); //app = append, out = output
  inFile.precision(6);
  outFile.precision(6);

  enum Axis {X, Y, Z};
  Axis axis;
  if (argv[2][0] == 'x' || argv[2][0] == 'X')
    axis = X;
  else if (argv[2][0] == 'y' || argv[2][0] == 'Y')
    axis = Y;
  else if (argv[2][0] == 'z' || argv[2][0] == 'Z')
    axis = Z;
  else {
    fprintf(stderr," axis not recognized! It should be x, y, or z.\n");
    exit(-1);
  }

  double deg = atof(argv[3]);

  char word1[20], word2[20];
  int integer1;
  double x,y,z,theta;
  double xi1, xi2;

  // load, process and output the node set
  int nNodes;
  inFile >> word1 >> word2;
  outFile << word1 << " " << word2 << endl;

  while(!inFile.eof()) {
    inFile >> word1;
    integer1 = strtol(word1,NULL,10);
    if(integer1==0) {outFile << word1; break;}
    inFile >> x >> y >> z;

    // project to (xi1, xi2)...
    if (axis == X) {
      xi1 = y;
      xi2 = z;
    } else if (axis == Y) {
      xi1 = x;
      xi2 = z;
    } else if (axis == Z) {
      xi1 = x;
      xi2 = y;
    }

    // rotate...
    double dist = sqrt(xi1*xi1 + xi2*xi2);
    if (dist>1e-16) {
      theta = asin(xi2/dist);
      if(theta>=0 && xi1<0)
        theta = PI - theta;
      else if(theta<0) {
        theta = 2.0*PI + theta;
        if (xi1<0)
          theta = 3.0*PI - theta;
      }
    }
    // now theta in [0, 2*Pi)
    theta += deg/180.0*PI;
    
    xi1 = dist*cos(theta);
    xi2 = dist*sin(theta);
 
    // get back to (x,y,z)
    if (axis == X) {
      y = xi1;
      z = xi2;
    } else if (axis == Y) {
      x = xi1;
      z = xi2;
    } else if (axis == Z) {
      x = xi1;
      y = xi2;
    }

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
