#include<iostream>
#include<fstream>
#include<stdio.h>
#include<stdlib.h>
#include<list>
using namespace std;

int main(int argc, char* argv[])
{
  if(argc!=3){
    fprintf(stderr,"Syntax: [binary] <orig. optDec file> <max elem id>\n");
    exit(-1);
  }
 
  ifstream input(argv[1],ios::in);
  ofstream output("output.optDec",ios::out);
  int Nmax = atoi(argv[2]);

  string line;
  getline(input,line);
  output << line << endl;

  int nSubs = 0;
  input >> nSubs;
  cout << "Decomposition: " << nSubs << " subdomains.\n";
  output << nSubs << endl;

  list<int> elems;
  int id(0), totalElems_orig(0), totalElems_new(0);
  for(int iSub=0; iSub<nSubs; iSub++) {
    elems.clear();
    int nElems_orig(0);
    input >> nElems_orig;
    totalElems_orig += nElems_orig;
    for(int i=0; i<nElems_orig; i++) {
      input >> id;
      if(id>Nmax) continue;
      elems.push_back(id); 
    } 
    totalElems_new += elems.size();
    output << elems.size() << "\n";
    for(auto it=elems.begin(); it!=elems.end(); it++)
      output << *it << "\n";
    cout << "SubDomain " << iSub+1 << ": " << nElems_orig << " elements --> " << elems.size() << " elements.\n";
  }
  cout << "Total: " << totalElems_orig << " elements --> " << totalElems_new << " elements.\n";

  input.close();
  output.close();
  return 0;
}
