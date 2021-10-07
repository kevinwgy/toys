#include<iostream>
#include<iomanip>
#include<cfloat> //DBL_MAX
#include<fstream>
#include<sstream>
#include<cstring>
#include<stdlib.h> //strtol
#include<map>
#include<vector>
#include"Vector3D.h"
#include"Vector2D.h"
using namespace std;

/** Project a 3D solution field with cylindrical symmetry to a 2D solution file
 *  The mesh file should be in the "top" format, start with the node list
 *  Each solution file should start with a line like "Scalar (or Vector) xx under load for xx"
 *  The next line should be the number of nodes. The third line should be the time solution is taken (not used)
 *  Starting the fourth line should be the actual solution.
 */

int main(int argc, char* argv[])
{
  if(argc<11) {
    cout << "Usage: <binary> [origin (x,y,z)] [axial dir (x,y,z)] [input 3D mesh file]\n";
    cout << "       [input solution file 1] [input solution file 2 (optional)] ... [input solution file N (optional)\n";
    cout << "       [time for output (or word 'last') (required)] [output file (required)]\n";
    exit(-1);
  }

  // origin and axis
  Vec3D x0(atof(argv[1]),atof(argv[2]),atof(argv[3]));
  Vec3D dir(atof(argv[4]),atof(argv[5]),atof(argv[6]));
  dir /= dir.norm();

  // files
  ifstream mesh(argv[7],ios::in);
  ofstream out(argv[argc-1],ios::out);
 
  int numSolFiles = argc - 10;
  ifstream sol[numSolFiles];
  for(int i=0; i<numSolFiles; i++)
    sol[i].open(argv[8+i],ios::in);

  // figure out which solution snapshot to output
  bool last_frame = false;
  double time_to_write = -1.0;
  if(!strcmp(argv[argc-2], "last") || !strcmp(argv[argc-2], "Last") || !strcmp(argv[argc-2], "LAST")) {
    last_frame = true;
    cout << "-- Will extract and output the last solution snapshot.\n";
  } else {
    time_to_write = atof(argv[argc-2]);
    cout << "-- Will extract and output the solution at time t = " << time_to_write << ".\n";
  } 
  
  // Read the mesh file and convert coords to 2D
  map<int,Vec2D> nodes;
  map<int,Vec3D> nodes3d;
  double xmin = DBL_MAX, xmax = -DBL_MAX, rmin = DBL_MAX, rmax = -DBL_MAX;

  string word, line;
  getline(mesh, line);
  istringstream iss(line);
  iss >> word;

  // make sure this is the node set
  if(word.compare(0,5,"Nodes",0,5) && word.compare(0,5,"NODES",0,5) && word.compare(0,5,"nodes",0,5)) {
    cout << "Error: Expecting word 'Nodes' on the first line of " << argv[7] << "." << endl;
    exit(-1);
  }
 
  while(!mesh.eof()) {
    mesh >> word;
    int id = strtol(word.c_str(),NULL,10);
    if(id==0) break;
    Vec3D x;
    mesh >> x[0] >> x[1] >> x[2];
    double a, b;
    a = (x - x0)*dir;
    b = (x - x0 - a*dir).norm();
    nodes[id] = Vec2D(a,b);
    nodes3d[id] = x;
    if(a<xmin) xmin = a;
    if(a>xmax) xmax = a;
    if(b<rmin) rmin = b;
    if(b>rmax) rmax = b;
  } 
  int nMeshNodes = nodes.size();
  cout << "-- Read " << nMeshNodes << " nodes from mesh file.\n";
  cout << "-- Bounding box (2D): (" << xmin << ", " << rmin << ") -> (" << xmax << ", " << rmax << ").\n";

  // Read the header of each solution file
  int solType[numSolFiles]; //1~scalar, 3~vector
  string solField[numSolFiles];
  int nSolNodes = 0;
  for(int i = 0; i<numSolFiles; i++) {
    getline(sol[i], line);
    istringstream iss2(line);
    iss2 >> word;
    if(!(word.compare(0,6,"Vector",0,6) && 
         word.compare(0,6,"VECTOR",0,6) && 
         word.compare(0,6,"vector",0,6)))
      solType[i] = 3;
    else if(!(word.compare(0,6,"Scalar",0,6) && 
              word.compare(0,6,"SCALAR",0,6) && 
              word.compare(0,6,"scalar",0,6)))
      solType[i] = 1;
    else {
      cout << "Error: Cannot understand solution file " << argv[8+i] << ". Expecting Vector or Scalar.\n";
      exit(-1);
    }

    // try to figure out the solution field;
    iss2 >> word;
    if(!(word.compare(0,3,"rho",0,3) && 
         word.compare(0,3,"Rho",0,3) && 
         word.compare(0,3,"RHO",0,3) && 
         word.compare(0,3,"den",0,3) && 
         word.compare(0,3,"DEN",0,3) && 
         word.compare(0,3,"Den",0,3))) {
      solField[i] = "Density";
      cout << "-- Solution field '" << word << "' is interpretted as the Density field.\n";
    } else if(!(word.compare(0,1,"v",0,1) && 
                word.compare(0,1,"V",0,1))) {
      solField[i] = "AxialVelocity  RadialVelocity";
      cout << "-- Solution field '" << word << "' is interpretted as the Velocity field (axial and radial).\n";
    } else if(!(word.compare(0,1,"p",0,1) && 
                word.compare(0,1,"P",0,1))) {
      solField[i] = "Pressure";
      cout << "-- Solution field '" << word << "' is interpretted as the Pressure field.\n";
    } else if(!(word.compare(0,1,"l",0,1) && 
                word.compare(0,1,"L",0,1))) {
      solField[i] = "LevelSet";
      cout << "-- Solution field '" << word << "' is interpretted as the LevelSet field.\n";
    } else if(!(word.compare(0,1,"m",0,1) && 
                word.compare(0,1,"M",0,1))) {
      solField[i] = "MaterialID";
      cout << "-- Solution field '" << word << "' is interpretted as the MaterialID field.\n";
    } else if(!(word.compare(0,1,"t",0,1) && 
                word.compare(0,1,"T",0,1))) {
      solField[i] = "Temperature";
      cout << "-- Solution field '" << word << "' is interpretted as the Temperature field.\n";
    } else {
      solField[i] = word;
      cout << "-- Solution field '" << word << "' is not translated. (May need to be updated in the output file manually).\n";
    }

    // read line #2. Should have the number of nodes in this solution file
    int n;
    sol[i] >> n;
    if(i==0)
      nSolNodes = n;
    else if(n != nSolNodes) {
      cout << "Error: numbers of data points in different solution files do not match. (" << nSolNodes << " vs. " << n << ").\n";
      exit(-1);
    }
    sol[i].ignore(256,'\n');
    
  }
  if(nSolNodes > nMeshNodes) {
    cout << "Error: Found " << nSolNodes << " in solution files but only " << nMeshNodes << " in the mesh file.\n"; 
    exit(-1);
  } else if (nSolNodes < nMeshNodes)
    cout << "Warning: Found " << nSolNodes << " in solution files but " << nMeshNodes << " in the mesh file.\n"; 
  

  if(last_frame) 
    out << "## 2D solution generated by program '3Dto2DProjection'. (Solution taken at the last time stamp.)" << endl;
  else
    out << "## 2D solution generated by program '3Dto2DProjection'. Time = " << time_to_write << "." << endl;

  out << "## GeneralCylindrical" << endl;
  out.precision(9);
  out << "## " << scientific << x0[0] << "  " << x0[1] << "  " << x0[2] << endl;
  out << "## " << scientific << dir[0] << "  " << dir[1] << "  " << dir[2] << endl;
  out << "## " << scientific << xmin << "  " << xmax << "  " << rmin << "  " << rmax << endl;
  out << "## AxialCoordinate  RadialCoordinate";
  for(int i=0; i<numSolFiles; i++)
    out << "  " << solField[i];
  out << endl;
  

  // Read solutions and extract the requested snapshot
  cout << "-- Reading the solution files.\n";
  vector<double> V[numSolFiles];
  double t;
  for(int i=0; i<numSolFiles; i++) { 

    if(solType[i]==1) V[i].resize(nSolNodes);
    else if(solType[i]==3) V[i].resize(3*nSolNodes);
    else {cout << "Error: Unknown solution type (solType[" << i << "] = " << solType[i] << "." << endl; exit(-1);}

    while(1) {
      sol[i] >> t;
      if(sol[i].eof()) {
        if(!last_frame) {
          cout << "Error: Failed to extract solution at t = " << time_to_write << " from " << argv[8+i] << ".\n";
          exit(-1);
        } else {
          cout << "-- Extracted last solution snapshot from " << argv[8+i] << ".\n";
        }
        break;
      }
      if(solType[i]==1) 
        for(int j=0; j<nSolNodes; j++)
          sol[i] >> V[i][j];
      else
        for(int j=0; j<nSolNodes; j++)
          sol[i] >> V[i][3*j] >> V[i][3*j+1] >> V[i][3*j+2];
      if(t == time_to_write) {
        cout << "-- Extracted solution at t = " << time_to_write << " from " << argv[8+i] << ".\n";
        break;
      }
    }
  } 

  cout << "-- Outputing solution at time t = " << t << "." << endl;
  out.precision(9);
  for(int i=0; i<nSolNodes; i++) {
    out << setw(20) << scientific << nodes[i+1][0] << setw(20) << scientific << nodes[i+1][1];
    for(int j=0; j<numSolFiles; j++) {
      if(solType[j] == 1) {
        double v;
        out << setw(20) << scientific << V[j][i];
      } else if (solType[j] == 3) {
        Vec3D v(V[j][3*i],V[j][3*i+1],V[j][3*i+2]);
        double a, b;
        a = v*dir;
        Vec3D dir2 = nodes3d[i+1] - x0 - nodes[i+1][0]*dir;
        if(dir2.norm()>0)
          dir2 /= dir2.norm();
        b = (v - a*dir)*dir2;
        out << setw(20) << scientific << a << setw(20) << scientific << b;
      }
    }
    out << endl;
  }

  // Cleanup
  out.close();
  mesh.close();
  for(int i=0; i<numSolFiles; i++)
    sol[i].close();   

  cout << "-- Done!" << endl;

  return 0;
}
