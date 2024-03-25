//-----------------------------------------------------------
//  Discription: Cut out a 2D slice of a geometric entity in 3D (stl, obj, or top format)
//        Usage: <input file> <x0> <y0> <z0> (origin on the cut-plane)
//               <nx> <ny> <nz> (normal direction) <output file (top)>
//       Author: Kevin Wang (created in Feb, 2024) 
//-----------------------------------------------------------

#include <iostream>
#include <iomanip> //setw
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <set>
#include <cstring>
#include <deque>
#include <array>
#include <list>
#include <map>
#include "GeoToolsLite.h"
#include "Vector2D.h"

using namespace std;
using namespace GeoTools;

void ReadMeshFileInTopFormat(const char *filename, vector<Vec3D> &Xs, vector<Int3> &Es);
void ReadMeshFileInOBJFormat(const char *filename, vector<Vec3D> &Xs, vector<Int3> &Es);
void ReadMeshFileInSTLFormat(const char *filename, vector<Vec3D> &Xs, vector<Int3> &Es);
void AddEdge(Vec3D X1, Vec3D X2, vector<Vec3D> &Xcut, vector<Int2> &Ecut);
bool same_strings_insensitive(std::string str1, std::string str2);

int verbose = 1;

// --------------------
// Main
// --------------------
int main(int argc, char* argv[]) {

  if (argc != 9) {
    std::cerr << "Usage: " << argv[0] << " <input file (STL, OBJ, or TOP)> <x0> <y0> <z0> (origin on "
                << "the cut-plane) <nx> <ny> <nz> (normal direction) <output file (TOP)>" << endl;
    exit(-1); 
  }

  // ------------------------------------------
  // Step 1: Read the input file
  // ------------------------------------------
  vector<Vec3D> Xs;
  vector<Int3> Es;
  
  string fname(argv[1]);
  auto loc = fname.find_last_of(".");
  if(loc>=fname.size()-1)//assume the default format (top) if file extension not detected
    ReadMeshFileInTopFormat(argv[1], Xs, Es);
  else {
    string ext = fname.substr(loc+1);
    if(same_strings_insensitive(ext, "obj"))
      ReadMeshFileInOBJFormat(argv[1], Xs, Es);
    else if(same_strings_insensitive(ext, "stl"))
      ReadMeshFileInSTLFormat(argv[1], Xs, Es);
    else
      ReadMeshFileInTopFormat(argv[1], Xs, Es);
  }
  fprintf(stdout,"- Found %d triangular elements in %s.\n", (int)Es.size(), argv[1]);



  // ------------------------------------------
  // Step 2: Slice! 
  // ------------------------------------------
  Vec3D O(atof(argv[2]), atof(argv[3]), atof(argv[4]));
  Vec3D dir(atof(argv[5]), atof(argv[6]), atof(argv[7]));
  assert(dir.norm()>0);
  dir /= dir.norm();
  fprintf(stdout,"- Slicing plane: Origin: (%e, %e, %e), Normal: (%e, %e, %e).\n",
          O[0], O[1], O[2], dir[0], dir[1], dir[2]);

  Vec3D P01, P12, P20;

  vector<Vec3D> Xcut;
  vector<Int2> Ecut;
  
  double plane_thickness = 1.0e-8; //should be larger than the tolerance for intersections

  for(auto&& elem : Es) {

    Vec3D& X0(Xs[elem[0]]);
    Vec3D& X1(Xs[elem[1]]);
    Vec3D& X2(Xs[elem[2]]);
    
    //First, check if the element is on the plane (within thickness)
    double d0 = ProjectPointToPlane(X0, O, dir, true);
    double d1 = ProjectPointToPlane(X1, O, dir, true);
    double d2 = ProjectPointToPlane(X2, O, dir, true);
    if(fabs(d0)<plane_thickness && fabs(d1)<plane_thickness && fabs(d2)<plane_thickness) {
      AddEdge(X0, X1, Xcut, Ecut);
      AddEdge(X1, X2, Xcut, Ecut);
      AddEdge(X2, X0, Xcut, Ecut);
      continue;
    }

    //Find intersections
    bool inter01 = LineSegmentIntersectsPlane(X0, X1, O, dir, NULL, &P01, true);
    bool inter12 = LineSegmentIntersectsPlane(X1, X2, O, dir, NULL, &P12, true);
    bool inter20 = LineSegmentIntersectsPlane(X2, X0, O, dir, NULL, &P20, true);
    if(inter01 && inter12)
      AddEdge(P01, P12, Xcut, Ecut);
    if(inter12 && inter20)
      AddEdge(P12, P20, Xcut, Ecut);
    if(inter20 && inter01)
      AddEdge(P20, P01, Xcut, Ecut);

  }  


  if(Ecut.size()==0) { //got nothing
    fprintf(stdout,"- Did not find any intersections between the input geometry and the slicing plane.\n");
    return 0;
  } else
    fprintf(stdout,"- Obtained %d nodes and %d edges/elements on the cut-plane.\n", Xcut.size(), Ecut.size());

  
  // ------------------------------------------
  // Step 3: Merge duplicate nodes
  // ------------------------------------------
  double dup_tolerance = plane_thickness*10.0; //larger than plane thickness to avoid tiny elements
  int N = Xcut.size();
  assert(N == 2*Ecut.size());
  vector<Vec3D> Xcut_cleaned;
  vector<int> nodemap(N, -1);
  for(int i=0; i<N; i++) { 
    bool found = false;
    for(int j=0; j<(int)Xcut_cleaned.size(); j++) { //not efficient, but should be fine
      if((Xcut_cleaned[j]-Xcut[i]).norm()<dup_tolerance) { //found duplicate
        nodemap[i] = j;
        found = true;
        break;
      }
    }
    if(!found) {
      Xcut_cleaned.push_back(Xcut[i]);
      nodemap[i] = Xcut_cleaned.size()-1; 
    }
  }
  for(auto&& elem : Ecut) {
    elem[0] = nodemap[elem[0]];
    elem[1] = nodemap[elem[1]]; 
  }
  fprintf(stdout,"- Detected and removed %d duplicate nodes. Tolerance: %e.\n", N-Xcut_cleaned.size(), dup_tolerance);


  // ------------------------------------------
  // Step 4: Drop trivial elements (single-node and duplicates)
  // ------------------------------------------
  list<Int2> Ecut_cleaned_list;
  set<int> hanging;
  for(auto&& elem : Ecut) {
    if(elem[0] != elem[1])
      Ecut_cleaned_list.push_back(elem);
    else
      hanging.insert(elem[0]); //possibly hanging
  }
  fprintf(stdout,"- Detected and removed %d degenerate elements (single node).\n",
                 Ecut.size()-Ecut_cleaned_list.size());
  

  Ecut_cleaned_list.sort([](Int2 a, Int2 b) {return a[0]+a[1] < b[0]+b[1];});
  int nDupElem = 0;
  { //put "it" inside a scope
    auto it = Ecut_cleaned_list.begin();
    while (it != Ecut_cleaned_list.end()) {
      int sum = (*it)[0] + (*it)[1];
      bool erased_elem = false;
      for(auto it2 = std::next(it); it2 != Ecut_cleaned_list.end(); it2++) {
        if ((*it2)[0] + (*it2)[1] != sum) //Ecut_cleaned_list is sorted by sum
          break;
        // sum is the same
        if((*it)[0] == (*it2)[0] || (*it)[0] == (*it2)[1]) {
          it = Ecut_cleaned_list.erase(it);
          erased_elem = true;
          nDupElem++;
          break;
        }
      }
      if(!erased_elem) //otherwise, "it" has already advanced.
        it++;
    }
  }
  fprintf(stdout,"- Detected and removed %d duplicated elements.\n", nDupElem);

  vector<Int2> Ecut_cleaned(Ecut_cleaned_list.begin(), Ecut_cleaned_list.end());

  // ------------------------------------------
  // Step 5: Find and drop hanging nodes
  // ------------------------------------------
  if(!hanging.empty()) {
    for(auto&& elem : Ecut_cleaned) {
      auto it = hanging.find(elem[0]);
      if(it!=hanging.end())
        hanging.erase(it); //not hanging
      auto it2 = hanging.find(elem[1]);
      if(it2!=hanging.end())
        hanging.erase(it2); //not hangin

      if(hanging.empty())
        break; //no need to keep checking
    } 
  }
  if(!hanging.empty()) {//these are really hanging nodes, drop them
    Xcut.clear();
    nodemap.assign(Xcut_cleaned.size(),-1);
    for(int i=0; i<(int)Xcut_cleaned.size(); i++) {
      if(hanging.find(i) == hanging.end()) {
        Xcut.push_back(Xcut_cleaned[i]);
        nodemap[i] = Xcut.size()-1;
      }
    }
    Xcut_cleaned = Xcut;
    for(auto&& elem : Ecut_cleaned) {
      elem[0] = nodemap[elem[0]];
      elem[1] = nodemap[elem[1]];
    }
  }
  fprintf(stdout,"- Dropped %d hanging nodes.\n", hanging.size());
  fprintf(stdout,"- Final: %d nodes, %d edges/elements.\n", Xcut_cleaned.size(), Ecut_cleaned.size());



  // ------------------------------------------
  // Step 6: Output the 2D slice in 3D space
  // ------------------------------------------
  std::fstream out;
  out.open(argv[8], std::fstream::out);
  if(!out.is_open()) {
    fprintf(stderr,"*** Error: Cannot write file %s.\n", argv[8]);
    exit(-1);
  }

  out << "Nodes MyNodes" << endl;
  for(int i=0; i<(int)Xcut_cleaned.size(); i++)
    out << std::setw(10) << i+1
        << std::setw(14) << std::scientific << Xcut_cleaned[i][0]
        << std::setw(14) << std::scientific << Xcut_cleaned[i][1]
        << std::setw(14) << std::scientific << Xcut_cleaned[i][2] << "\n";
  out << "Elements MyElems using MyNodes" << endl;
  for(int i=0; i<(int)Ecut_cleaned.size(); i++)
    out << std::setw(10) << i+1 << "  1  "  //"1" for line segment
        << std::setw(10) << Ecut_cleaned[i][0]+1
        << std::setw(10) << Ecut_cleaned[i][1]+1 << "\n";

  out.flush();
  out.close();
  fprintf(stdout,"- Wrote the 2D slice (in the original 3D space) to %s.\n", argv[8]);



  // ------------------------------------------
  // Step 7: Output the 2D slice in 2D
  // ------------------------------------------
  Vec3D Xi, Eta; //axes
  GetOrthonormalVectors(dir, Xi, Eta, true);
  fprintf(stdout,"- Created 2D coordinate system: O(%e, %e, %e), X(%e, %e, %e), Y(%e, %e, %e).\n",
          O[0], O[1], O[2], Xi[0], Xi[1], Xi[2], Eta[0], Eta[1], Eta[2]);
  fname = string(argv[8]);
  loc = fname.find_last_of(".");
  if(loc>=fname.size()-1)//file extension not detected
    fname = fname + "_xy";
  else 
    fname.insert(loc, "_xy");
  out.open(fname.c_str(), std::fstream::out);
  if(!out.is_open()) {
    fprintf(stderr,"*** Error: Cannot write file %s.\n", fname.c_str());
    exit(-1);
  }

  out << "Nodes MyNodesXY" << endl;
  for(int i=0; i<(int)Xcut_cleaned.size(); i++) {
    double xi  = (Xcut_cleaned[i] - O)*Xi;
    double eta = (Xcut_cleaned[i] - O)*Eta;
    out << std::setw(10) << i+1
        << std::setw(14) << std::scientific << xi
        << std::setw(14) << std::scientific << eta
        << std::setw(14) << "0.0\n";
  }
  out << "Elements MyElemsXY using MyNodesXY" << endl;
  for(int i=0; i<(int)Ecut_cleaned.size(); i++)
    out << std::setw(10) << i+1 << "  1  "  //"1" for line segment
        << std::setw(10) << Ecut_cleaned[i][0]+1
        << std::setw(10) << Ecut_cleaned[i][1]+1 << "\n";

  out.flush();
  out.close();
  fprintf(stdout,"- Wrote the 2D slice (using the new 2D coordinates) to %s.\n", fname.c_str());

  fprintf(stdout,"- Successful completion.\n");

  return 0;
}


//------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------

void
ReadMeshFileInTopFormat(const char *filename, vector<Vec3D> &Xs, vector<Int3> &Es)
{

  // read data from the surface input file.
  FILE *topFile;
  topFile = fopen(filename, "r");
  if(topFile == NULL) {
    fprintf(stderr,"*** Error: Cannot open embedded surface mesh file (%s).\n", filename);
    exit(-1);
  }
 
  int MAXLINE = 500;
  char line[MAXLINE], key1[MAXLINE], key2[MAXLINE]; //, copyForType[MAXLINE];

  int num0 = 0;
  int num1 = 0;
  double x1, x2, x3;
  int node1, node2, node3;
  int type_reading = 0; //1 means reading node set; 2 means reading element set
  std::deque<std::pair<int, Vec3D>> nodeList;
  std::deque<std::array<int, 4>> elemList; // element ID + three node IDs
  int maxNode = 0, maxElem = 0;
  bool found_nodes = false;
  bool found_elems = false;


  // --------------------
  // Read the file
  // --------------------
  while(fgets(line, MAXLINE, topFile) != 0) {

    sscanf(line, "%s", key1);
    string key1_string(key1);

    if(key1[0] == '#') {
      //Do nothing. This is user's comment
    }
    else if(same_strings_insensitive(key1_string,"Nodes")){
      if(found_nodes) {//already found nodes... This is a conflict
        fprintf(stderr,"*** Error: Found multiple sets of nodes (keyword 'Nodes') in %s.\n", filename);
        exit(-1);
      }
      sscanf(line, "%*s %s", key2);
      type_reading = 1;
      found_nodes = true;
    }
    else if(same_strings_insensitive(key1_string, "Elements")) {

      if(found_elems) {//already found elements... This is a conflict
        fprintf(stderr,"*** Error: Found multiple sets of elements (keyword 'Elements') in %s.\n", filename);
        exit(-1);
      }
      type_reading = 2;
      found_elems = true;

    }
    else if(type_reading == 1) { //reading a node (following "Nodes Blabla")
      int count = sscanf(line, "%d %lf %lf %lf", &num1, &x1, &x2, &x3);
      if(count != 4) {
        fprintf(stderr,"*** Error: Cannot interpret line %s (in %s). Expecting a node.\n", line, filename);
        exit(-1);
      }
      if(num1 < 1) {
        fprintf(stderr,"*** Error: detected a node with index %d in embedded surface file %s.\n", num1, filename);
        exit(-1);
      }
      if(num1 > maxNode)
        maxNode = num1;

      nodeList.push_back({num1, {x1, x2, x3}});
    }
    else if(type_reading == 2) { // we are reading an element --- HAS TO BE A TRIANGLE!
      int count = sscanf(line, "%d %d %d %d %d", &num0, &num1, &node1, &node2, &node3);
      if(count != 5) {
        fprintf(stderr,"*** Error: Cannot interpret line %s (in %s). Expecting a triangular element.\n", line, filename);
        exit(-1);
      }
      if(num0 < 1) {
        fprintf(stderr,"*** Error: detected an element with index %d in embedded surface file %s.\n", num0, filename);
        exit(-1);
      }
      if(num0 > maxElem)
        maxElem = num0;

      elemList.push_back({num0, node1, node2, node3});
    }
    else { // found something I cannot understand...
      fprintf(stderr,"*** Error: Unable to interpret line %s (in %s).\n", line, filename);
      exit(-1);
    }

  }

  fclose(topFile);

  if(!found_nodes) {
    fprintf(stderr,"*** Error: Unable to find node set in %s.\n", filename);
    exit(-1);
  }
  if(!found_elems) {
    fprintf(stderr,"*** Error: Unable to find element set in %s.\n", filename);
    exit(-1);
  }

  // ----------------------------
  // Now, check and store nodes
  // ----------------------------
  int nNodes = nodeList.size();
  map<int,int> old2new;
  Xs.resize(nNodes);
  int id(-1);
  if(nNodes != maxNode) { // need to renumber nodes, i.e. create "old2new"
    fprintf(stdout,"Warning: The node indices of an embedded surface may have a gap: "
                  "max index = %d, number of nodes = %d. Renumbering nodes. (%s)\n",
                  maxNode, nNodes, filename);
//    assert(nNodes < maxNode);

    int current_id = 0; 
    vector<bool> nodecheck(maxNode+1, false);
    for(auto it1 = nodeList.begin(); it1 != nodeList.end(); it1++) {
      id = it1->first;
      if(nodecheck[id]) {
        fprintf(stderr,"*** Error: Found duplicate node (id: %d) in embedded surface file %s.\n", id, filename);
        exit(-1);
      }
      nodecheck[id] = true;
      Xs[current_id] = it1->second; 
      old2new[id] = current_id;
      current_id++;
    }
    assert(current_id==(int)Xs.size());
  } 
  else { //in good shape
    vector<bool> nodecheck(nNodes, false);
    for(auto it1 = nodeList.begin(); it1 != nodeList.end(); it1++) {
      id = it1->first - 1; 
      if(nodecheck[id]) {
        fprintf(stderr,"*** Error: Found duplicate node (id: %d) in embedded surface file %s.\n", id+1, filename);
        exit(-1);
      }
      nodecheck[id] = true;
      Xs[it1->first - 1] = it1->second;
    }
  }


  // ------------------------------
  // check nodes used by elements
  // ------------------------------
  for(auto it = elemList.begin(); it != elemList.end(); it++) {

    id = (*it)[0];
    node1 = (*it)[1];
    node2 = (*it)[2];
    node3 = (*it)[3];
      
    if(old2new.empty()) {//node set is original order

      if(node1<=0 || node1 > nNodes) {
        fprintf(stderr,"*** Error: Detected unknown node number (%d) in element %d (%s).\n", node1, id, filename);
        exit(-1);
      }

      if(node2<=0 || node2 > nNodes) {
        fprintf(stderr,"*** Error: Detected unknown node number (%d) in element %d (%s).\n", node2, id, filename);
        exit(-1);
      }

      if(node3<=0 || node3 > nNodes) {
        fprintf(stderr,"*** Error: Detected unknown node number (%d) in element %d (%s).\n", node3, id, filename);
        exit(-1);
      }
    }
    else {// nodes are renumbered

      auto p1 = old2new.find(node1);
      if(p1 == old2new.end()) { 
        fprintf(stderr,"*** Error: Detected unknown node number (%d) in element %d (%s).\n", node1, id, filename);
        exit(-1);
      }

      auto p2 = old2new.find(node2);
      if(p2 == old2new.end()) { 
        fprintf(stderr,"*** Error: Detected unknown node number (%d) in element %d (%s).\n", node2, id, filename);
        exit(-1);
      }

      auto p3 = old2new.find(node3);
      if(p3 == old2new.end()) { 
        fprintf(stderr,"*** Error: Detected unknown node number (%d) in element %d (%s).\n", node3, id, filename);
        exit(-1);
      }
    }
  }


  // ----------------------------
  // check and store elements
  // ----------------------------
  int nElems = elemList.size();
  Es.resize(nElems);
  if(nElems != maxElem) { // need to renumber elements.
    fprintf(stdout,"Warning: The element indices of an embedded surface may have a gap: "
                  "max index = %d, number of elements = %d. Renumbering elements. (%s)\n",
                  maxElem, nElems, filename);
//    assert(nElems < maxElem);
    
    int current_id = 0; 
    vector<bool> elemcheck(maxElem+1, false);
    for(auto it = elemList.begin(); it != elemList.end(); it++) {
      id = (*it)[0];
      if(elemcheck[id]) {
        fprintf(stderr,"*** Error: Found duplicate element (id: %d) in embedded surface file %s.\n", id, filename);
        exit(-1);
      }
      elemcheck[id] = true;

      node1 = (*it)[1];
      node2 = (*it)[2];
      node3 = (*it)[3];
      
      if(old2new.empty()) //node set is original order
        Es[current_id] = Int3(node1-1, node2-1, node3-1);
      else {// nodes are renumbered
        auto p1 = old2new.find(node1);
        auto p2 = old2new.find(node2);
        auto p3 = old2new.find(node3);
        Es[current_id] = Int3(p1->second, p2->second, p3->second);
      }      
      current_id++;
    }
  } 
  else { //element numbers in good shape

    vector<bool> elemcheck(nElems, false);
    for(auto it = elemList.begin(); it != elemList.end(); it++) {
      id = (*it)[0] - 1;
      if(elemcheck[id]) {
        fprintf(stderr,"*** Error: Found duplicate element (id: %d) in embedded surface file %s.\n", id, filename);
        exit(-1);
      }
      elemcheck[id] = true;

      node1 = (*it)[1];
      node2 = (*it)[2];
      node3 = (*it)[3];
      
      if(old2new.empty()) //node set is original order
        Es[id] = Int3(node1-1, node2-1, node3-1);
      else {// nodes are renumbered
        auto p1 = old2new.find(node1);
        auto p2 = old2new.find(node2);
        auto p3 = old2new.find(node3);
        Es[id] = Int3(p1->second, p2->second, p3->second);
      }
    }
  }

}


//------------------------------------------------------------------------------------------------

void
ReadMeshFileInOBJFormat(const char *filename, vector<Vec3D> &Xs, vector<Int3> &Es)
{
  // This function is adapted from toys::obj2top. But unlike the "toy", it does not separate different groups

  Xs.clear();
  Es.clear();

  std::ifstream input(filename, std::ifstream::in);
  if(input.fail()) {
    fprintf(stderr,"*** Error: Cannot open embedded surface mesh file (%s).\n", filename);
    exit(-1);
  }

  string line, word;

  double area_tol = 1e-12;

  Vec3D xyz;
  int nVerts, maxVerts = 1024, n1, n2, n3;
  vector<int> ind(maxVerts); //should not have polygons with more than 1024 vertices!
  vector<std::pair<string, vector<Int3> > > elem_groups;
  vector<Int3> *elems = NULL; 

  std::set<string> ignored_keywords;

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
      Xs.push_back(xyz);
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
                      filename, maxVerts); 
          exit(-1); 
        }
      }
      if(nVerts<3) {
        fprintf(stderr,"*** Error: Found a face element in %s with only %d nodes.\n",
                    filename, nVerts); 
        exit(-1); 
      }
      for(int i=1; i<nVerts-1; i++)
        elems->push_back(Int3(ind[0], ind[i], ind[i+1]));
    }
    else
      ignored_keywords.insert(word);
  }


  for(auto&& key : ignored_keywords) {
    if(!key.empty())
      fprintf(stdout,"Warning: Ignored lines in %s starting with %s.\n",
                    filename, key.c_str());
  }

  if(verbose>=1) {
    fprintf(stdout,"- Found %d nodes in %s.\n", (int)Xs.size(), filename);
    for(auto&& eg : elem_groups)
      fprintf(stdout,"- Obtained %d triangular elements in group %s from %s.\n",
            (int)eg.second.size(), eg.first.c_str(), filename);

    if(elem_groups.size()>=2)
      fprintf(stdout,"Warning: Merging multiple (%d) groups into one.\n", (int)elem_groups.size()); 
  }


  for(auto&& eg : elem_groups) {
    for(auto&& e : eg.second) {
      n1 = e[0]; n2 = e[1]; n3 = e[2];
      if(n1<=0 || n1>(int)Xs.size() || n2<=0 || n2>(int)Xs.size() ||
         n3<=0 || n3>(int)Xs.size()) {
        fprintf(stderr,"*** Error: Found element (%d %d %d) in %s with unknown node(s).\n",
                n1, n2, n3, filename);
        exit(-1);
      }
      Vec3D cr = (Xs[n2-1] - Xs[n1-1])^(Xs[n3-1] - Xs[n1-1]);
      if(cr.norm() < area_tol) {
        fprintf(stdout,"Warning: Detected a degenerate triangle with area %e --- dropped from the list.\n",
                      cr.norm());
        continue;
      }
      Es.push_back(Int3(n1-1,n2-1,n3-1)); //node id starts at 0
    }
  }

  input.close();
}


//------------------------------------------------------------------------------------------------

void
ReadMeshFileInSTLFormat(const char *filename, vector<Vec3D> &Xs, vector<Int3> &Es)
{
  // This function is adapted from toys::stl2top.

  Xs.clear();
  Es.clear();
  
  std::ifstream input(filename, std::ifstream::in);

  string line;
  string word;
  getline(input, line);

  double area_tol = 1e-12;

  int nodeid(0);
  double x,y,z;
 
  while(true) {

    getline(input, line);

    if(line.compare(0,8,"endsolid") == 0)
      break;

    getline(input, line);
    input >> word >> x >> y >> z;
    Vec3D X1(x,y,z);
    input >> word >> x >> y >> z;
    Vec3D X2(x,y,z);
    input >> word >> x >> y >> z;
    Vec3D X3(x,y,z);
    Vec3D cr = (X2 - X1)^(X3 - X1);
    getline(input, line); //get the end of line
    getline(input, line);
    getline(input, line);

    if(cr.norm() < area_tol) {
      fprintf(stdout,"Warning: Detected a degenerate triangle with area %e --- dropped from the list.\n",
                    cr.norm());
      continue;
    }

    Xs.push_back(X1);
    Xs.push_back(X2);
    Xs.push_back(X3);
    Es.push_back(Int3(nodeid, nodeid+1, nodeid+2));
    nodeid += 3;

  }

  input.close();
}

//------------------------------------------------------------------------------------------------

void AddEdge(Vec3D X1, Vec3D X2, vector<Vec3D> &Xcut, vector<Int2> &Ecut)
{
  int N = Xcut.size();
  Xcut.push_back(X1);
  Xcut.push_back(X2);
  Ecut.push_back(Int2(N,N+1));
}

//------------------------------------------------------------------------------------------------
bool same_strings_insensitive(std::string str1, std::string str2)
{
  return ((str1.size() == str2.size()) &&
           std::equal(str1.begin(), str1.end(), str2.begin(),
                      [](char &c1, char &c2){return (c1==c2 || std::toupper(c1)==std::toupper(c2));}));
}
