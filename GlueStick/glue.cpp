//-----------------------------------------------------------
//  Discription: Combine multiple mesh files and solution files for visualization
//               <number of frames> <step> <reference scalar value>
//               <path to mesh file 1> <solution file 1> 
//               <path to mesh file 2> <solution file 2> 
//               (optional) <path to mesh file 3> <solution file 3>
//      Outputs: new mesh file -- output.top 
//               new sol  file -- output.sol
//               new disp file -- output.disp
//       Author: Kevin Wang (created on Aug.13,2010)
//        Notes: 1. This software currently handles only 2 or 3 meshes/solutions
//                  but it's easy to modified it to handle more. 
//               2. This software assumes all solutions have the same time-stepping.
//               3. (WARNING) this code doesn't verify every input argument.
//               4. (WARNING) this code assumes that the input files
//                            have the "normal" format.
//-----------------------------------------------------------

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>
using namespace std;
ofstream topFile("output.top",ios::out);
ofstream solFile("output.sol",ios::out);
ofstream dispFile("output.disp",ios::out);
ifstream mesh[3];
ifstream sol[3];
#define BAR 1
int main (int argc, char* argv[])
{
  // *****************************************
  // Element type map (id -> number of nodes)
  map<int,int> eMap;
  eMap[1]   = 2;
  eMap[4]   = 3;
  eMap[5]   = 4;
  eMap[106] = 2;
  eMap[104] = 3;
  eMap[108] = 3;
  eMap[188] = 4;
  // *****************************************

  if((argc!=8)&&(argc!=10)) {
    fprintf(stderr,"Incorrect usage!\n");
    fprintf(stderr,"Format: [binary] <number of frames> <step> <reference scalar value>\n");
    fprintf(stderr,"                 <path to mesh file 1> <solution file 1> \n");
    fprintf(stderr,"                 <path to mesh file 2> <solution file 2> \n");
    fprintf(stderr,"                 (optional) <path to mesh file 3> <solution file 3>\n");
    exit(-1);
  }

  char word1[30], word2[30], word3[30], word4[30], word5[30], word6[30];
  int i1, i2, i3, i4;
  double x, y, z;
  int count = 0;
  int Total = (argc==8) ? 2 : 3;
  int step  = atoi(argv[2]);

  // *****************************************
  //       FIRST WE HANDLE THE MESHES
  // *****************************************
  map<int,int> Id[Total];

  // load mesh1 nodes  and dump to output.top
  mesh[0].open(argv[4],ios::in);
  mesh[0] >> word1 >> word2;
  topFile << "Nodes UnitedNodes" << endl;
  while(!mesh[0].eof()) {
    mesh[0] >> word1;
    i1 = strtol(word1,NULL,10);
    if(i1==0)
      break;
            
    Id[0][i1] = ++count;
    mesh[0] >> x >> y >> z;
    topFile << count << " " << x << " " << y << " " << z << endl;
  }
  fprintf(stderr,"Loaded %d nodes from Mesh 1.\n",count);

  // load mesh2 nodes  and dump to output.top
  int ddd = count;
  mesh[1].open(argv[6],ios::in);
  mesh[1] >> word1 >> word2;
  while(!mesh[1].eof()) {
    mesh[1] >> word1;
    i1 = strtol(word1,NULL,10);
    if(i1==0)
      break;

    Id[1][i1] = ++count;
    mesh[1] >> x >> y >> z;
    topFile << count << " " << x << " " << y << " " << z << endl;
  }
  fprintf(stderr,"Loaded %d nodes from Mesh 2.\n",count-ddd);

  // load mesh3 nodes  and dump to output.top
  if(argc==10) {
    mesh[2].open(argv[8],ios::in);
    ddd = count;
    mesh[2] >> word1 >> word2;
    while(!mesh[2].eof()) {
      mesh[2] >> word1;
      i1 = strtol(word1,NULL,10);
      if(i1==0)
        break;

      Id[2][i1] = ++count;
      mesh[2] >> x >> y >> z;
      topFile << count << " " << x << " " << y << " " << z << endl;
    }
    fprintf(stderr,"Loaded %d nodes from Mesh 3.\n",count-ddd);
  }

  count = 0; //clear
  // load mesh1 elems and dump to output.top
  mesh[0] >> word1 >> word2 >> word3;
  topFile << "Elements UnitedSol using UnitedNodes" << endl;
  while(!mesh[0].eof()) {
    mesh[0] >> word1;
    if(mesh[0].eof())
      break;
    i1 = strtol(word1,NULL,10);
    if(i1==0)
      break;
    topFile << ++count << " ";  // element Id

    mesh[0] >> word1;
    i1 = strtol(word1,NULL,10);
    topFile << i1; // element type
    map<int,int>::iterator it = eMap.find(i1);
    if(it==eMap.end()) {
      fprintf(stderr,"Couldn't find Element type %d. Please add it to the list (embedded in the code).\n", i1); 
      exit(-1);
    }
    
    int nNodes = it->second; // # nodes in this element.
    for(int i=0; i<nNodes; i++) {
      mesh[0] >> i1;
      it = Id[0].find(i1);
      if(it==Id[0].end()) {
        cout << " Couldn't find Node " << i1 << "in Mesh 1. (It's used in elements!)" << endl;
        exit(-1);
      }
      topFile << " " << it->second;
    }
    topFile << endl;
  }
  fprintf(stderr,"Loaded %d elements from Mesh 1.\n", count);
  mesh[0].close();

  // load mesh2 elems and dump to output.top
  ddd = count;
  mesh[1] >> word1 >> word2 >> word3;
  while(!mesh[1].eof()) {
    mesh[1] >> word1;
    if(mesh[1].eof())
      break;
    i1 = strtol(word1,NULL,10);
    if(i1==0)
      break;
    topFile << ++count << " ";  // element Id

    mesh[1] >> word1;
    i1 = strtol(word1,NULL,10);
    topFile << i1; // element type
    map<int,int>::iterator it = eMap.find(i1);
    if(it==eMap.end()) {
      fprintf(stderr,"Couldn't find Element type %d. Please add it to the list (embedded in the code).\n", i1);
      exit(-1);
    }
    
    int nNodes = it->second; // # nodes in this element.
    int myCode = i1;
#ifdef BAR
    int bar[4];
#endif
    for(int i=0; i<nNodes; i++) {
      mesh[1] >> i1;
      it = Id[1].find(i1);
      if(it==Id[1].end()) {
        cout << " Couldn't find Node " << i1 << "in Mesh 2. (It's used in elements!)" << endl;
        exit(-1);
      }
      topFile << " " << it->second;
#ifdef BAR
      if(myCode==188)
        bar[i] = it->second;
#endif
    }
    topFile << endl;
#ifdef BAR
    if(myCode==188) {
      topFile << ++count << " " << (int)106 << " " << bar[0] << " " << bar[1] << endl;
      topFile << ++count << " " << (int)106 << " " << bar[1] << " " << bar[2] << endl;
      topFile << ++count << " " << (int)106 << " " << bar[2] << " " << bar[3] << endl;
      topFile << ++count << " " << (int)106 << " " << bar[3] << " " << bar[0] << endl;
    }
#endif
  }
  fprintf(stderr,"Loaded %d elements from Mesh 2.\n", count-ddd);
  mesh[1].close();

  // load mesh3 elems and dump to output.top
  if(argc==10) {
    ddd = count;
    mesh[2] >> word1 >> word2 >> word3;
    while(!mesh[2].eof()) {
      mesh[2] >> word1;
      if(mesh[2].eof())
        break;
      i1 = strtol(word1,NULL,10);
      if(i1==0)
        break;
      topFile << ++count << " ";  // element Id

      mesh[2] >> word1;
      i1 = strtol(word1,NULL,10);
      topFile << i1; // element type
      map<int,int>::iterator it = eMap.find(i1);
      if(it==eMap.end()) {
        fprintf(stderr,"Couldn't find Element type %d. Please add it to the list (embedded in the code).\n", i1);
        exit(-1);
      }

      int nNodes = it->second; // # nodes in this element.
      for(int i=0; i<nNodes; i++) {
        mesh[2] >> i1;
        it = Id[2].find(i1);
        if(it==Id[2].end()) {
          cout << " Couldn't find Node " << i1 << "in Mesh 3. (It's used in elements!)" << endl;
          exit(-1);
        }
        topFile << " " << it->second;
      }
      topFile << endl;
    }
    fprintf(stderr,"Loaded %d elements from Mesh 3.\n", count-ddd);
    mesh[1].close();
  }

  topFile.close();


  // *****************************************
  //        NOW WORK ON SOLUTIONS
  // *****************************************
  count = 0;
  int nFrames = atoi(argv[1]);
  double u0 = atof(argv[3]);
  int nCol[Total], nData[Total];
  int nTotNodes = 0, maxData = 0, maxCol = 0;
  
  // prepare solutions 
  for(int i=0; i<Total; i++) {
    sol[i].open(argv[5+2*i], ios::in);
    sol[i] >> word1 >> word2 >> word3 >> word4 >> word5 >> word6;
    if(word1[0]=='V')  nCol[i] = 3;  else if(word1[0]=='S') nCol[i] = 1;
    else {fprintf(stderr,"Error in solution %d! Can't figure out if it's scalar or vector!\n",i+1);exit(-1);}
    sol[i] >> word1;
    nData[i] = atoi(word1);
    nTotNodes += nData[i];
    maxData = max(maxData, nData[i]);
    maxCol = max(maxCol, nCol[i]);
  }

  // prepare outputs
  fprintf(stderr,"Preparing to generate %d frames of combined solution(scalar and vector) on %d nodes\n", nFrames, nTotNodes);
  solFile << "Scalar scalar under load for UnitedNodes" << endl;
  solFile << nTotNodes << endl;
  dispFile << "Vector DISP under load for UnitedNodes" << endl;
  dispFile << nTotNodes << endl;

  double u[maxCol];
  int skip = 0;
  for(int iFrame=0; iFrame<nFrames; iFrame++) {

    if((++skip)==step)
      skip = 0;

    for(int i=0; i<Total; i++) {
      sol[i] >> word1; //time
      if(i==0) 
        if(!skip) {
          solFile << word1 << endl;
          dispFile << word1 << endl;
        }

      for(int j=0; j<nData[i]; j++) { 
        for(int k=0; k<nCol[i]; k++) 
          sol[i] >> u[k];
        if(skip)
          continue;
        if(nCol[i]==1) {
          solFile << u[0] << endl;
          //solFile << scientific << u[0] << endl;
          dispFile << "0.0 0.0 0.0" << endl;
        } else if(nCol[i]==3) {
          solFile << u0 << endl;
          dispFile << u[0] << " " <<  u[1] << " " << u[2] << endl;
          //dispFile << scientific << u[0] << " " <<  u[1] << " " << u[2] << endl;
        }
      }
    }
    if(!skip)
      fprintf(stderr,"Done with Frame %d (Total = %d)\r", iFrame+1, nFrames);
  }
  fprintf(stderr,"\n");
  fprintf(stderr,"Done!\n");
 
  for(int i=0; i<Total; i++)
    sol[i].close();

  solFile.close();
  dispFile.close();
  
  return 0;
}
