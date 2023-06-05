#include<iostream>
#include<fstream>
#include<cassert>
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

  double area_tol = 1e-12;

  Vec3D xyz;
  int nVerts, maxVerts = 1024, n1, n2, n3;
  vector<int> ind(maxVerts); //should not have polygons with more than 1024 vertices!
  vector<Vec3D> nodes;
  vector<std::pair<string, vector<Int3> > > elem_groups;
  vector<Int3> *elems = NULL; 

  set<string> ignored_keywords;

  int line_number = 0;

  while(getline(input, line)) {

    line_number++;

    auto first_nonspace_id = line.find_first_not_of(" ");
    if((unsigned)first_nonspace_id<line.size() && line[first_nonspace_id] == '#')
      continue;  //this line is comment

    std::stringstream linestream(line);
    if(!(linestream >> word)) //the first word in the line
      continue;

    if(word == "v") { //vertex
      linestream >> xyz[0] >> xyz[1] >> xyz[2]; //skipping vertex properties (if present)
      nodes.push_back(xyz);
    }
    else if(word == "g") { //element group
      string group_name;
      linestream >> group_name; //the second word in the line -> group name  
      while(linestream >> word) //in stl, group name may have multiple words...
        group_name = group_name + "_" + word;

      bool found(false);
      for(auto&& eg : elem_groups) {
        if(eg.first == group_name) {
          elems = &eg.second;
          found = true;
          break;
        }
      }
      if(!found) {
        elem_groups.push_back(std::make_pair(group_name, vector<Int3>()));
        elems = &elem_groups.back().second;
      }
    }
    else if(word == "f") { //face element

      if(elem_groups.empty()) { //user did not specify group name
        assert(elems==NULL);
        elem_groups.push_back(std::make_pair("Default", vector<Int3>()));
        elems = &elem_groups.back().second;
      }

      nVerts = 0;
      while(linestream >> word) { 
        ind[nVerts++] = std::stoi(word.substr(0,word.find("/")));
        if(nVerts>maxVerts) {
          fprintf(stderr,"*** Error: Found a face element in %s with more than %d nodes.\n",
                  argv[1], maxVerts); 
          fprintf(stderr,"%s\n", line.c_str());
          exit(-1); 
        }
      }
      if(nVerts<3) {
        fprintf(stderr,"*** Error: Found a face element in %s with only %d nodes.\n",
                argv[1], nVerts); 
        exit(-1); 
      }
      for(int i=1; i<nVerts-1; i++)
        elems->push_back(Int3(ind[0], ind[i], ind[i+1]));
    }
    else {
      if(word=="50475/51567/202020") fprintf(stdout,"GOT YOU!!!! Line: %d\n", line_number);
      ignored_keywords.insert(word);
    }
  }


  for(auto&& key : ignored_keywords) {
    if(!key.empty())
      fprintf(stderr,"Warning: Ignored lines in %s starting with %s.\n",
              argv[1], key.c_str());
  }

  fprintf(stdout, "Found %d nodes in %s.\n", (int)nodes.size(), argv[1]);
  for(auto&& eg : elem_groups)
    fprintf(stdout, "Obtained %d triangular elements in group %s from %s.\n",
            (int)eg.second.size(), eg.first.c_str(), argv[1]);


  // Writing the mesh
  output << "Nodes SurfaceNodes\n";
  int counter = 0; 
  for(auto it = nodes.begin(); it != nodes.end(); it++)
    output << std::setw(12) << ++counter << std::setw(20) << std::scientific << (*it)[0]
           << std::setw(20) << std::scientific << (*it)[1] << std::setw(20) << std::scientific << (*it)[2] << "\n";

  output << "Elements Surface using SurfaceNodes\n";
  counter = 0;
  for(auto&& eg : elem_groups) {
    elems = &eg.second;
    for(auto it = elems->begin(); it != elems->end(); it++) {
      //check if area is non-zero
      n1 = (*it)[0]; n2 = (*it)[1]; n3 = (*it)[2];
      if(n1<=0 || n1>(int)nodes.size() || n2<=0 || n2>(int)nodes.size() ||
         n3<=0 || n3>(int)nodes.size()) {
        fprintf(stderr,"*** Error: Found element (%d %d %d) with unknown node(s).\n",
                n1, n2, n3);
        exit(-1);
      }
      Vec3D cr = (nodes[n2-1] - nodes[n1-1])^(nodes[n3-1] - nodes[n1-1]);
      if(cr.norm() < area_tol) {
        fprintf(stderr,"Warning: Detected a degenerate triangle with area %e --- dropped from the list.\n",
        cr.norm());
        continue;
      }
      output << std::setw(12) << ++ counter << "    4" << std::setw(12) << n1
             << std::setw(12) << n2 << std::setw(12) << n3  << "\n";
    }
  }

  int nGroups = 0; //count the number of non-empty groups
  for(auto&& eg : elem_groups) {
    if(eg.second.empty())
      continue;
    nGroups++;
  }

  // print surface components / groups
  if(nGroups>1) {
    for(auto&& eg : elem_groups) {
      if(eg.second.empty())
        continue;
      output << "Elements " << eg.first << " using SurfaceNodes\n";
      counter = 0;
      elems = &eg.second;
      for(auto it = elems->begin(); it != elems->end(); it++) {
        //check if area is non-zero
        n1 = (*it)[0]; n2 = (*it)[1]; n3 = (*it)[2];
        Vec3D cr = (nodes[n2-1] - nodes[n1-1])^(nodes[n3-1] - nodes[n1-1]);
        if(cr.norm() < area_tol)
          continue;
        output << std::setw(12) << ++ counter << "    4" << std::setw(12) << n1
               << std::setw(12) << n2 << std::setw(12) << n3 << "\n";
      }
    }
  }

  input.close();
  output.close();
  return 0;
}
