//-----------------------------------------------------------
//  Discription: This routine connects Mesh 1 and Mesh 3.
//       Inputs: <matching of 1 and 2> <matching of 1 and 3> 
//      Outputs: <matched> <nonmatched> (based on mesh 1)
//       Author: Kevin Wang (Jun.3,2010)
//        Notes: (WARNING) this code doesn't verify inputs.
//-----------------------------------------------------------
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <list>
using namespace std;

int main(int argc, char* argv[])
{
  //parsing
  if(argc!=3) { 
    cout<<"Format: [binary] <matching of 1 and 2> <matching of 1 and 3>" << endl;
    exit(-1);
  }
  ifstream mesh1(argv[1],ios::in);
  ifstream mesh2(argv[2],ios::in);
  ofstream matched("Output.top",ios::out);

  //global variables
  int nNodes1, nNodes2, i1, i2;
  list<pair<int,int> > matching1;
  list<pair<int,int> > matching2;
  list<pair<int,int> >::iterator it;

  //load the matching between 1 and 2
  char word1[50];
  double x,y,z;
  nNodes1 = 0;
  while(!mesh1.eof()) {
    word1[0]=0;
    mesh1 >> word1;
    i1 = strtol(word1,NULL,10);
    if(i1==0) break;
    mesh1 >> i2 >> x >> y >> z;
    matching1.push_back(pair<int,int>(i1,i2));
  }
  nNodes1 = matching1.size();
  int Match1[nNodes1][2];
  int count = 0;
  for(it=matching1.begin();it!=matching1.end();it++){
    Match1[count][0] = it->first;
    Match1[count][1] = it->second;
    count++;
  }
  mesh1.close();
  matching1.clear();

  //load the matching between 1 and 3
  while(!mesh2.eof()) {
    word1[0]=0;
    mesh2 >> word1;
    i1 = strtol(word1,NULL,10);
    if(i1==0) break;
    mesh2 >> i2 >> x >> y >> z;
    matching2.push_back(pair<int,int>(i1,i2));
  } 
  nNodes2 = matching2.size();
  int Match2[nNodes2][2];
  count = 0;
  for(it=matching2.begin();it!=matching2.end();it++){
    Match2[count][0] = it->first;
    Match2[count][1] = it->second;
    count++;
  }
  mesh2.close();
  matching2.clear();
  
  //write output: the new matching.
  int current = 0;
  int a1, a2;
  bool found = false;
  for(int i=0; i<nNodes1; i++){
    found = false;
    a1 = Match1[i][0];
    a2 = Match1[i][1];
    for(int j=current; j<nNodes2; j++) 
      if(Match2[j][0]==a1) {
        matched << a2 << " " << Match2[j][1] << scientific << " " << 0.0 << " " << 0.0 << " " << 0.0 << endl;
        current = j;
        found = true;
        break;
      }
    if(!found)
      for(int j=0; j<current; j++)
        if(Match2[j][0]==a1) {
          matched << a2 << " " << Match2[j][1] << scientific << " " << 0.0 << " " << 0.0 << " " << 0.0 << endl;
          current = j;
          found = true;
          break;
        }
    if(!found)
      fprintf(stderr,"Node %d in Mesh 1 is not matched!\n",a1);
  } 
  matched.close();
  fprintf(stderr,"Done\n");

  return 0;
}
