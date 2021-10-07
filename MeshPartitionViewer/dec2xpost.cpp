#include<iostream>
#include<fstream>
#include<stdio.h>
#include<stdlib.h>
#include<vector>
#include<limits>       // std::numeric_limits
#include<string>
using namespace std;

int main(int argc, char* argv[])
{
  if(argc!=3){
    fprintf(stderr,"Syntax: [binary] <mesh file> <partnmesh/partdmesh decomposition file>\n");
    exit(-1);
  }
 
  ifstream mesh(argv[1],ios::in);
  ifstream input(argv[2],ios::in);
  ofstream output("mesh_partition.xpost",ios::out);

  string line;
  getline(input,line);
  getline(input,line);
  cout << line << endl;

  // read the decomposition file
  int nPart;
  input >> nPart;
  vector<int> parts[nPart];
  int totalElems = 0, nMax = 0;

  for(int i=0; i<nPart; i++) {
    int nElems;
    input >> nElems;
    parts[i].resize(nElems);
    for(int j=0; j<nElems; j++) {
      input >> parts[i][j];
      if(parts[i][j]>nMax)
        nMax = parts[i][j];
    }
    totalElems += nElems;
    cout << "Read partition " << i << ": " << nElems << "nodes.\n";
  }

  cout << "Max Element Id: " << nMax << endl;
  if(nMax != totalElems) {
    cout << "Error: index gap or duplicated element(s). totalElems = " << totalElems << ", nMax = " << nMax << ".\n";
    exit(-1);
  }

  // create unified element list
  vector<int> allElems;
  allElems.resize(nMax, numeric_limits<int>::max());
  for(int i=0; i<nPart; i++)
     for(int j=0; j<parts[i].size(); j++)
       if(allElems[parts[i][j]-1] > i+1)
         allElems[parts[i][j]-1] = i+1;


  // read mesh and create output
  string NodesName;
  mesh >> line;
  mesh >> NodesName;
  getline(mesh,line);

  int nodeCounter = 0;
  while(true) {
    getline(mesh,line);
    if (line.compare(0,8,"Elements") == 0)
      break;
    nodeCounter++;
  }
  cout << "Found " << nodeCounter << " nodes in the mesh.\n";
  vector<int> nodes;
  nodes.resize(nodeCounter, numeric_limits<int>::max());

  int id, code, node, part;
  for(int i=0; i<nMax; i++) {
    mesh >> id >> code;
    if(code != 5) {
      cout << "Error: Currently only tetrahedron elements (code = 5) are supported. Can be easily updated. (detected code " << code << ").\n";
      exit(-1);
    }
    part = allElems[id-1];
    for(int j=0; j<4; j++) {
      mesh >> node;
      if(node>nodeCounter) {
        cout << "Error in node numbering. (" << node << " vs " << nodeCounter << ")." << endl;
        exit(-1);
      }
      if(part < nodes[node-1])
        nodes[node-1] = part;
    }
  }
  
  output << "Scalar subdomain under load for " << NodesName << endl;
  output << nodeCounter << endl;
  output << "0" << endl;
  for(int i=0; i<nodeCounter; i++)
    output << (double)(nodes[i]) << "\n";
  cout << "Done.\n";


  input.close();
  output.close();
  return 0;
}
