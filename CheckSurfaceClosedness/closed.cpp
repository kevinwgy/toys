//-----------------------------------------------------------
//  Discription: check how many times each edge is traversed 
//       Inputs: <path to input mesh file> 
//               <path to output file (edges traversed N times)>
//               <N> 
//      Outputs: new mesh file -- output.top 
//       Author: Kevin Wang (created on Feb.16,2012)
//        Notes: (WARNING) this code doesn't verify inputs.
//               (WARNING) this code assumes the mesh file
//                         starts with the node set
//-----------------------------------------------------------

#include<fstream>
#include<iostream>
#include<string>
#include<map>
using namespace std;

struct edge {
  int n1, n2;
  edge() {n1 = n2 = -1;}
  edge(int _n1, int _n2) {if(_n1<=_n2) {n1=_n1;n2=_n2;} else {n1=_n2;n2=_n1;}}
  ~edge() {}
  bool operator<(const edge & e2) const {
    if(this->n1<e2.n1)
      return true;
    else if (this->n1>e2.n1)
      return false;
    else if (this->n2<e2.n2)
      return true;
    else return false;
    return false;
  }
  edge &operator=(const edge &e2) {
    this->n1 = e2.n1;
    this->n2 = e2.n2;
    return *this;
  }
};

int main(int argc, char* argv[])
{
  if(argc!=4){
    cout << "Usage: [binary] <path to input mesh file> <path to output file (edges traversed N times)> <N>" << endl;
    exit(-1);
  }

  map<edge,int> edges;
  map<edge,int>::iterator it;

  ifstream infile(argv[1],ios::in);
  ofstream outfile(argv[2],ios::out);
  int id, a,b,c, code, count;
  id = -1; count = 0;
  char word1[100], word2[100], word3[100], word4[100];
  string line;
  edge e[3];

  while(!infile.eof()) {
    infile >> word1;
    if(word1[0] == 'E' && word1[1] == 'l' && word1[2] == 'e' && word1[3] == 'm') {
      infile >> word2 >> word3 >> word4;
      cout << "Checking " << word1 << " " << word2 << " " << word3 << " " << word4 << endl;
      break;
    } else getline(infile,line);
  }

  while(!infile.eof()) {
    word1[0] = '0';
    word1[1] = '\n';
    infile >> word1;
    id = strtol(word1,NULL,10);
    if(id==0) {break;}
    count++;
    infile >> code >> a >> b >> c;
    e[0] = edge(a,b);
    e[1] = edge(b,c);
    e[2] = edge(a,c);

    for(int i=0; i<3; i++) {
      it = edges.find(e[i]);
      if(it==edges.end()) {
        edges[e[i]] = 1;
      } else 
        it->second++;
    }
  }
  cout << "Loaded " << count << " elements." << endl;
  
  map<int,int> nCalls;
  map<int,int>::iterator it2;
  count = 0;
  outfile << "Elements EdgeSet" << argv[3] << " using " << word4 << endl;
  for(it=edges.begin(); it!=edges.end(); it++) {
    int nc = it->second;
    it2 = nCalls.find(nc);
    if(it2==nCalls.end())
      nCalls[nc] = 1;
    else
      nCalls[nc]++;
 
    if(nc==atoi(argv[3]))
      outfile << ++count << " 1 " << it->first.n1 << " " << it->first.n2 << endl;
  }

  for(it2=nCalls.begin(); it2!=nCalls.end(); it2++)
    cout << "Edges that are called " << it2->first << " times: " << it2->second << endl;
    
  infile.close();
  outfile.close();
  return 0;
}
