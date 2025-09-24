#include<cstdio>
#include<cstdlib>
#include<vector>
#include<map>
#include<array>
#include<string>
#include<cassert>
#include<deque>
#include<iostream>
#include<fstream>
#include<cstring>
#include<cstdarg>
#include<iomanip>
#include "Vector3D.h"
#include "KDTree.h"

using namespace std;

//----------------------------------------------------------------------------------------
//Utility functions copied from M2C
void print_error(const char format[],...);
void print_warning(const char format[],...);
void ReadMeshFileInTopFormat(const char *filename, vector<Vec3D> &Xs, vector<Int3> &Es);
bool same_strings_insensitive(std::string str1, std::string str2);
//----------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------
//This is a user-provided function that determines if a node ('x') should be deleted.
bool remove_node(Vec3D &x) {
  //if(x[1]<0.20 || (x[1]<0.236 && x[0]>-0.06))
  //  return true;
  return false;
}
//This is a user-provided function that determines if a triangle should be deleted.
bool remove_elem(Vec3D &x1, Vec3D &x2, Vec3D &x3) {
  if(x1[1]<0.2)
    return true;
  if(x1[1]<0.232 && x1[0]>-0.06)
    return true;
  if(x1[1]<0.236 && x1[0]>-0.06) {
    double d12 = (x1-x2).norm(), d23 = (x2-x3).norm(), d31 = (x3-x1).norm();
    double dmin = std::min(d12, std::min(d23, d31));
    double dmax = std::max(d12, std::max(d23, d31));
    if(dmin<0.0003 || dmax<0.00083)
      return true;
  }
  if(x1[0]>-0.0538 || x2[0]>-0.0538 || x3[0]>-0.0538)
    return true;
  if((x1[0]>-0.056 || x2[0]>-0.056 || x3[0]>-0.056) &&
     (x1[2]>-0.085 || x2[2]>-0.085 || x3[2]>-0.085))
    return true;

  return false;
}
//----------------------------------------------------------------------------------------


int main(int argc, char* argv[])
{
  if(argc!=2) {
    cout << "Usage: <executible file [CUSTOMIZED]> <path to mesh (top format)>" << endl;
    exit(-1);
  }

  vector<Vec3D> Xs;
  vector<Int3> Es;
  ReadMeshFileInTopFormat(argv[1], Xs, Es);

  int nNodes = Xs.size();
  int nElems = Es.size();

  // delete elements
  vector<bool> drop(nElems, false);
  vector<bool> active(nNodes, false);
  int count_dropped_e = 0;
  for(int i=0; i<nElems; i++) {
    if(remove_elem(Xs[Es[i][0]], Xs[Es[i][1]], Xs[Es[i][2]])) {
      drop[i] = true;
      count_dropped_e++;
    } else 
      active[Es[i][0]] = active[Es[i][1]] = active[Es[i][2]] = true;
  }

  fprintf(stdout,"- Detected %d elements that satisfied the element-based criterion.\n", count_dropped_e);
  int count_hanging = 0;
  for(int i=0; i<nNodes; i++)
    if(!active[i])
      count_hanging++;
  fprintf(stdout,"- Detected %d hanging nodes (likely due to element removal) --- will be deleted.\n", count_hanging);
   
  // now, figure out new id to trim surface
  vector<int> newid(nNodes, -1);
  int current_id = 0;
  int count_dropped_n = 0;
  for(int i=0; i<nNodes; i++) {
    if(!active[i])
      continue;
    if(remove_node(Xs[i])) {
      count_dropped_n++;
      continue;
    }
    newid[i] = current_id++;
  }
  fprintf(stdout,"- Detected %d (additional) nodes that satisfied the node-based criterion.\n", count_dropped_n);
  if(count_dropped_n==0 && count_dropped_e==0 && count_hanging==0) {
    fprintf(stdout,"- Nothing to do. Exit.\n");
    return 0;
  }


  // output cleaned mesh
  std::ofstream out("output.top", std::ios::out);
  if(!out.is_open()) {
    print_error("*** Error: Cannot write output file.\n");
    exit(-1);
  }
  out << "Nodes OutputNodes" << endl;
  for(int i=0; i<nNodes; i++) {
    if(newid[i]==-1)
      continue;
    out << std::setw(10) << newid[i]+1
        << std::setw(14) << std::scientific << Xs[i][0]
        << std::setw(14) << std::scientific << Xs[i][1] 
        << std::setw(14) << std::scientific << Xs[i][2] << "\n";
  }
  out << "Elements OutputElements using OutputNodes" << endl;
  current_id = 0;
  count_dropped_e = 0;
  for(int i=0; i<nElems; i++) {
    if(drop[i])
      continue;
    if(newid[Es[i][0]]==-1 || newid[Es[i][1]]==-1 || newid[Es[i][2]]==-1) {
      count_dropped_e++;
      continue;
    }
    out << std::setw(10) << ++current_id << "  4  " //"4" for triangles
        << std::setw(10) << newid[Es[i][0]]+1 
        << std::setw(10) << newid[Es[i][1]]+1 
        << std::setw(10) << newid[Es[i][2]]+1 << "\n";
  }
  fprintf(stdout,"- Deleted %d additional elements connected to deleted nodes.\n", count_dropped_e);

  out.flush();
  out.close();
  fprintf(stdout,"- Wrote output file 'output.top'.\n");
  fprintf(stdout,"- Done.\n");

  return 0;
}
//------------------------------------------------------------------------------------------
//END OF MAIN FUNCTION
//------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------
//Utility functions copied from m2c, with minor mods to avoid the need of MPI
void print_error(const char format[],...)
{
  char format_colored[strlen(format)+40];
  strcat(format_colored, "\033[0;31m");
  strcat(format_colored, format);
  strcat(format_colored, "\033[0m");

  va_list Argp;
  va_start(Argp, format);
  vprintf(format_colored, Argp);
    va_end(Argp);

  return;
}

//----------------------------------------------------------------------------------------

void print_warning(const char format[],...)
{
  char format_colored[strlen(format)+40];
  strcat(format_colored, "\033[0;35m");
  strcat(format_colored, format);
  strcat(format_colored, "\033[0m");

  va_list Argp;
  va_start(Argp, format);
  vprintf(format_colored, Argp);
  va_end(Argp);

  return;
}

//----------------------------------------------------------------------------------------

bool same_strings_insensitive(std::string str1, std::string str2)
{
  return ((str1.size() == str2.size()) && std::equal(str1.begin(), str1.end(), str2.begin(),
                      [](char &c1, char &c2){return (c1==c2 || std::toupper(c1)==std::toupper(c2));}));
}

//----------------------------------------------------------------------------------------

void
ReadMeshFileInTopFormat(const char *filename, vector<Vec3D> &Xs, vector<Int3> &Es)
{

  // read data from the surface input file.
  FILE *topFile;
  topFile = fopen(filename, "r");
  if(topFile == NULL) {
    print_error("*** Error: Cannot open embedded surface mesh file (%s).\n", filename);
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
        print_error("*** Error: Found multiple sets of nodes (keyword 'Nodes') in %s.\n", filename);
        exit(-1);
      }
      sscanf(line, "%*s %s", key2);
      type_reading = 1;
      found_nodes = true;
    }
    else if(same_strings_insensitive(key1_string, "Elements")) {

      if(found_elems) {//already found elements... This is a conflict
        print_error("*** Error: Found multiple sets of elements (keyword 'Elements') in %s.\n", filename);
        exit(-1);
      }
      type_reading = 2;
      found_elems = true;

    }
    else if(type_reading == 1) { //reading a node (following "Nodes Blabla")
      int count = sscanf(line, "%d %lf %lf %lf", &num1, &x1, &x2, &x3);
      if(count != 4) {
        print_error("*** Error: Cannot interpret line %s (in %s). Expecting a node.\n", line, filename);
        exit(-1);
      }
      if(num1 < 1) {
        print_error("*** Error: detected a node with index %d in embedded surface file %s.\n", num1, filename);
        exit(-1);
      }
      if(num1 > maxNode)
        maxNode = num1;

      nodeList.push_back({num1, {x1, x2, x3}});
    }
    else if(type_reading == 2) { // we are reading an element --- HAS TO BE A TRIANGLE!
      int count = sscanf(line, "%d %d %d %d %d", &num0, &num1, &node1, &node2, &node3);
      if(count != 5) {
        print_error("*** Error: Cannot interpret line %s (in %s). Expecting a triangular element.\n", line, filename);
        exit(-1);
      }
      if(num0 < 1) {
        print_error("*** Error: detected an element with index %d in embedded surface file %s.\n", num0, filename);
        exit(-1);
      }
      if(num0 > maxElem)
        maxElem = num0;

      elemList.push_back({num0, node1, node2, node3});
    }
    else { // found something I cannot understand...
      print_error("*** Error: Unable to interpret line %s (in %s).\n", line, filename);
      exit(-1);
    }

  }

  fclose(topFile);

  if(!found_nodes) {
    print_error("*** Error: Unable to find node set in %s.\n", filename);
    exit(-1);
  }
  if(!found_elems) {
    print_error("*** Error: Unable to find element set in %s.\n", filename);
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
    print_warning("Warning: The node indices of an embedded surface may have a gap: "
                  "max index = %d, number of nodes = %d. Renumbering nodes. (%s)\n",
                  maxNode, nNodes, filename);
//    assert(nNodes < maxNode);

    int current_id = 0; 
    vector<bool> nodecheck(maxNode+1, false);
    for(auto it1 = nodeList.begin(); it1 != nodeList.end(); it1++) {
      id = it1->first;
      if(nodecheck[id]) {
        print_error("*** Error: Found duplicate node (id: %d) in embedded surface file %s.\n", id, filename);
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
        print_error("*** Error: Found duplicate node (id: %d) in embedded surface file %s.\n", id+1, filename);
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
        print_error("*** Error: Detected unknown node number (%d) in element %d (%s).\n", node1, id, filename);
        exit(-1);
      }

      if(node2<=0 || node2 > nNodes) {
        print_error("*** Error: Detected unknown node number (%d) in element %d (%s).\n", node2, id, filename);
        exit(-1);
      }

      if(node3<=0 || node3 > nNodes) {
        print_error("*** Error: Detected unknown node number (%d) in element %d (%s).\n", node3, id, filename);
        exit(-1);
      }
    }
    else {// nodes are renumbered

      auto p1 = old2new.find(node1);
      if(p1 == old2new.end()) { 
        print_error("*** Error: Detected unknown node number (%d) in element %d (%s).\n", node1, id, filename);
        exit(-1);
      }

      auto p2 = old2new.find(node2);
      if(p2 == old2new.end()) { 
        print_error("*** Error: Detected unknown node number (%d) in element %d (%s).\n", node2, id, filename);
        exit(-1);
      }

      auto p3 = old2new.find(node3);
      if(p3 == old2new.end()) { 
        print_error("*** Error: Detected unknown node number (%d) in element %d (%s).\n", node3, id, filename);
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
    print_warning("Warning: The element indices of an embedded surface may have a gap: "
                  "max index = %d, number of elements = %d. Renumbering elements. (%s)\n",
                  maxElem, nElems, filename);
//    assert(nElems < maxElem);
    
    int current_id = 0; 
    vector<bool> elemcheck(maxElem+1, false);
    for(auto it = elemList.begin(); it != elemList.end(); it++) {
      id = (*it)[0];
      if(elemcheck[id]) {
        print_error("*** Error: Found duplicate element (id: %d) in embedded surface file %s.\n", id, filename);
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
        print_error("*** Error: Found duplicate element (id: %d) in embedded surface file %s.\n", id, filename);
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

//----------------------------------------------------------------------------------------

