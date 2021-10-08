//-----------------------------------------------------------
//  Discription: This code assembles the distributed data
//       Inputs: <# subD's> <number of data>
//               <path to nodeMap 1> <path to nodeMap 2> ...
//               <path to dataFile 1><path to dataFile 2> ...
//      Outputs: assembled data file
//       Author: Kevin Wang (created on Aug.25,2009)
//        Notes: (WARNING) this code doesn't verify inputs.
//               format of "nodeMap":
//               0 1
//               1 2
//               2 3
//               ...
//               Column 1: local index  (should start from 0)
//               Column 2: global index (should start from 1)
//               format of "dataFile"
//               0 1.5 2.8 3.7 ... 
//               1 2.3 4.6 4.6 ...
//               .................
//               Column 1: local index (should start from 0)
//               Column 2 ~ N: data
//                 N: number of data (input).
//-----------------------------------------------------------

#include<stdio.h>
#include<stdlib.h>
#include<list>
#include<math.h>

int main(int argc, char* argv[])
{
  // parse
  int nSub = atoi(argv[1]);
  int nData = atoi(argv[2]); 
  char* nodeFileName[nSub];
  char* dataFileName[nSub];
  for(int i=0; i<nSub; i++) {
    nodeFileName[i] = argv[i+3];
    dataFileName[i] = argv[3+nSub+i];
  }

  // setup files
  FILE* nodeFile[nSub];
  FILE* dataFile[nSub];
  FILE* outputFile;
  for (int i=0; i<nSub; i++) {
    nodeFile[i] = fopen(nodeFileName[i], "r");
    dataFile[i] = fopen(dataFileName[i], "r");
  }
  outputFile = fopen("output.sol", "w");

  // load node maps.
  std::list<std::pair<int, int> > locToGlob[nSub];
  std::list<std::pair<int, int> >::iterator it; 
  char c1[30];
  int num1, num2, nNodes[nSub], nGlobNodes;
  int *locToGlobMap[nSub];
  nGlobNodes = 0;

  for (int i=0; i<nSub; i++) { 
    while(1){
      int valid = fscanf(nodeFile[i],"%s", c1);
      if(valid!=1) break;
      num1 = strtol(c1, NULL, 10);
      fscanf(nodeFile[i], "%d\n", &num2);
      if(num2>nGlobNodes) 
        nGlobNodes = num2;
      locToGlob[i].push_back(std::pair<int,int>(num1, num2));               
    }
    nNodes[i] = locToGlob[i].size();
    locToGlobMap[i] = new int[nNodes[i]];

    for(it=locToGlob[i].begin(); it!=locToGlob[i].end(); it++)
      locToGlobMap[i][it->first] = it->second;

    fprintf(stderr,"Loaded NodeSet %d: %d nodes.\n", i+1, nNodes[i]);
    locToGlob[i].clear();
  }  
  fprintf(stderr,"Total number of nodes: %d\n", nGlobNodes);
     
  // load and merge datasets
  double globData[nGlobNodes][nData];
  double check[nData];
  bool filled[nGlobNodes];
  for(int i=0; i<nGlobNodes; i++)
    filled[i] = false;

  for (int i=0; i<nSub; i++) {
    int index;
    int globIndex;

    for (int j=0; j<nNodes[i]; j++) {

      bool merge = false;
      fscanf(dataFile[i], "%d", &index);
      globIndex = locToGlobMap[i][index]-1;

      if(!filled[globIndex])
        filled[globIndex] = true;
      else {
        merge = true;
        for (int iData=0; iData<nData; iData++)
          check[iData] = globData[globIndex][iData];
      }
        
      for(int iData=0; iData<nData-1; iData++)
        fscanf(dataFile[i], "%lf", &(globData[globIndex][iData]));
      fscanf(dataFile[i], "%lf\n", &(globData[globIndex][nData-1]));

      if(merge) { //check for inconsistency in dataSets
        double norm = 0;
        for (int iData=0; iData<nData; iData++)
          norm += (check[iData]-globData[globIndex][iData])*(check[iData]-globData[globIndex][iData]);
        norm = sqrt(norm);
        if(norm>=1.0e-5)
          fprintf(stderr,"WARNING: inconsistency at GlobNode %d\n", globIndex+1);
      }
    }
  } 

  // check for unfilled nodes
  int count = 0;
  for (int i=0; i<nGlobNodes; i++) 
    if(!filled[i]) count++;
  if(count>0)
    fprintf(stderr,"WARNING: on %d nodes data is not found.\n", count);

  // write output file
  for (int i=0; i<nGlobNodes; i++) {
    for (int j=0; j<nData-1; j++)
      fprintf(outputFile, "%.8e ", globData[i][j]);
    fprintf(outputFile,"%.8e\n", globData[i][nData-1]);
  }

  // clean up.
  for (int i=0; i<nSub; i++) {
    fclose(nodeFile[i]);
    fclose(dataFile[i]);
    delete[] locToGlobMap[i];
  }
  fclose(outputFile);

  return 0;
}
