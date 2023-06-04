#include<iostream>
#include<fstream>
#include<string>
#include<iomanip> //setw
#include<vector>
#include<set>
#include"Vector3D.h"
using std::vector;
using std::set;
using std::string;

int main(int argc, char* argv[])
{

  if(argc!=3) {
    fprintf(stderr,"input format: [binary] <path to obj file (input)> <path to top file (output)>\n");
    exit(-1);
  }

  std::ifstream input(argv[1], std::ifstream::in);
  std::ofstream output(argv[2], std::ofstream::out);

  string line;
  string word;
//  getline(input, line);

  Vec3D xyz;
  int nVerts, maxVerts = 100, n1, n2, n3;
  vector<int> ind(maxVerts); //should not have polygons with more than 100 vertices!
  vector<Vec3D> nodes;
  vector<Int3> elems;
 
  set<string> ignored_keywords;

  while(getline(input, line)) {

    auto first_nonspace_id = line.find_first_not_of(" ");
    if((unsigned)first_nonspace_id<line.size() && line[first_nonspace_id] == '#')
      continue;  //this line is comment

    std::stringstream linestream(line);
    linestream >> word; //the first word in the line

    if(word == "v") { //vertex
      linestream >> xyz[0] >> xyz[1] >> xyz[2]; //skipping vertex properties (if present)
      nodes.push_back(xyz);
    }
    else if(word == "f") { //face element
      nVerts = 0;
      while(linestream >> word) { 
        ind[nVerts++] = std::stoi(word.substr(0,word.find("/")));
        if(nVerts>maxVerts) {
          fprintf(stderr,"*** Error: Found a face element in %s with more than %d nodes.\n",
                  argv[1], maxVerts); 
          exit(-1); 
        }
      }
      if(nVerts<3) {
        fprintf(stderr,"*** Error: Found a face element in %s with only %d nodes.\n",
                argv[1], nVerts); 
        exit(-1); 
      }
      for(int i=1; i<nVerts-1; i++)
        elems.push_back(Int3(ind[0], ind[i], ind[i+1]));
    }
    else
      ignored_keywords.insert(word);
  }

  for(auto&& key : ignored_keywords)
    fprintf(stderr,"Warning: Ignored contents in %s starting with keyword %s.\n",
            argv[1], key.c_str());
  fprintf(stdout, "Obtained %d nodes and %d triangular elements from %s.\n",
          (int)nodes.size(), (int)elems.size(), argv[1]);

  // Writing the mesh
  output << "Nodes SurfaceNodes\n";
  int counter = 0; 
  for(auto it = nodes.begin(); it != nodes.end(); it++)
    output << std::setw(12) << ++counter << std::setw(20) << std::scientific << (*it)[0]
           << std::setw(20) << std::scientific << (*it)[1] << std::setw(20) << std::scientific << (*it)[2] << "\n";

  output << "Elements Surface using SurfaceNodes\n";
  counter = 0;
  vector<Vec3D> nodes_v(nodes.begin(), nodes.end());
  for(auto it = elems.begin(); it != elems.end(); it++) {
    //check if area is non-zero
    n1 = (*it)[0]; n2 = (*it)[1]; n3 = (*it)[2];
    Vec3D cr = (nodes_v[n2] - nodes_v[n1])^(nodes_v[n3] - nodes_v[n1]);
    if(cr.norm() < 1e-8) {
      fprintf(stderr,"Warning: Detected a degenerate triangle --- dropped from the list.\n");
      continue;
    }
    output << std::setw(12) << ++ counter << "    4" << std::setw(12) << (*it)[0]+1 
           << std::setw(12) << (*it)[1]+1 << std::setw(12) << (*it)[2]+1 << "\n";
  }

  input.close();
  output.close();
  return 0;
}
