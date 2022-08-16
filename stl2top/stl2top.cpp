#include<iostream>
#include<fstream>
#include<string>
#include<list>
#include<iomanip> //setw
#include<vector>
#include"Vector3D.h"
using namespace std;

int main(int argc, char* argv[])
{

  if(argc!=3) {
    fprintf(stderr,"input format: [binary] <path to stl file (input)> <path to top file (output)>\n");
    exit(-1);
  }

  ifstream input(argv[1], ios::in);
  ofstream output(argv[2], ios::out);


  string line;
  string word;
  getline(input, line);

  int nodeid(0);
  int n1, n2, n3;
  double x,y,z;
  int i1, i2, i3;
  list<Vec3D> nodes;
  list<Int3> elems;
  int iter = 0;
 
  while(true) {

    iter++;

    getline(input, line);

//    cout << iter << "  " <<  line << endl;

    if(line.compare(0,8,"endsolid") == 0)
      break;

    getline(input, line);
    input >> word >> x >> y >> z;
    nodes.push_back(Vec3D(x,y,z));
    input >> word >> x >> y >> z;
    nodes.push_back(Vec3D(x,y,z));
    input >> word >> x >> y >> z;
    nodes.push_back(Vec3D(x,y,z));
    elems.push_back(Int3(nodeid, nodeid+1, nodeid+2));
    nodeid += 3;

    getline(input, line); //get the end of line
    getline(input, line);
    getline(input, line);
  }

  cout << "Found " << elems.size() << " triangular elements.\n";

  output << "Nodes SurfaceNodes" << endl;
  int counter = 0; 
  for(auto it = nodes.begin(); it != nodes.end(); it++)
    output << std::setw(12) << ++counter << std::setw(20) << std::scientific << (*it)[0]
           << std::setw(20) << std::scientific << (*it)[1] << std::setw(20) << std::scientific << (*it)[2] << "\n";

  output << "Elements Surface using SurfaceNodes" << endl;
  counter = 0;
  vector<Vec3D> nodes_v(nodes.begin(), nodes.end());
  for(auto it = elems.begin(); it != elems.end(); it++) {
    //check if area is non-zero
    n1 = (*it)[0]; n2 = (*it)[1]; n3 = (*it)[2];
    Vec3D cr = (nodes_v[n2] - nodes_v[n1])^(nodes_v[n3] - nodes_v[n1]);
    if(cr.norm() < 1e-8) {
      cerr << "*** Detected a degenerate triangle --- dropped from the list." << endl;
      continue;
    }
    output << std::setw(12) << ++ counter << "    4" << std::setw(12) << (*it)[0]+1 
           << std::setw(12) << (*it)[1]+1 << std::setw(12) << (*it)[2]+1 << "\n";
  }

  input.close();
  output.close();
  return 0;
}
