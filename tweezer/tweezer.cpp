//-----------------------------------------------------------
//  Discription: This routine finds the neighbors of a specified
//               node and print out a local mesh for display 
//       Inputs: <path to mesh file> <# nodes> <# elements> <node #>
//      Outputs: local mesh file -- localMesh.top
//       Author: Kevin Wang (created on Jan.1,2009)
//        Notes: (WARNING) this code doesn't verify inputs.
//-----------------------------------------------------------

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <list>
#include <set>
using namespace std;

struct Elem {
  int id, type, a, b, c, d;
  Elem(int id_, int type_, int a_, int b_, int c_, int d_) {
    id = id_, type = type_; a = a_; b = b_; c = c_; d = d_;
  }
};

int main(int argc, char* argv[])
{
  if(argc!=5) {
    fprintf(stderr,"input format: [binary] <path to mesh file> <# nodes> <# elements> <node #>\n");
    exit(-1);
  }
  int nGNodes = atoi(argv[2]);
  int nGElems = atoi(argv[3]);

  int sample = atoi(argv[4]);

  double X[nGNodes][3], x[3];
  FILE* source = 0, *output = 0;
  char non1[30], non2[30], non3[30], non4[30], non5[30]; 
  int ff;

// load mesh.
  source = fopen(argv[1],"r");
  if (!source) fprintf(stderr,"ERROR. can't find the original mesh.\n");
  output = fopen("localMesh.top","w");
  if (!output) fprintf(stderr,"ERROR. can't establish the output file.\n");
  fscanf(source,"%s %s\n", non1, non2);
  fprintf(output,"%s %s\n", non1, "LocalNodes");
  for (int i=0; i<nGNodes; i++) {
    fscanf(source,"%d %lf %lf %lf\n", &ff, &(x[0]), &(x[1]), &(x[2]));
    for(int j=0; j<3; j++)
      X[ff-1][j] = x[j];
  }
  fprintf(stderr,"DONE with nodes.\n");

  list<Elem> elems;
  set<int> nodes;
  fscanf(source,"%s %s %s %s\n", non1, non2, non3, non4);
  int d1, d2, d3, d4, d5, d6;
  for (int i=0; i<nGElems; i++) {
    fscanf(source, "%d %d %d %d %d %d\n", &d1, &d2, &d3, &d4, &d5, &d6);
    if (d3==sample || d4==sample || d5==sample || d6==sample ) {
      nodes.insert(d3);
      nodes.insert(d4);
      nodes.insert(d5);
      nodes.insert(d6);
      Elem myElem(d1,d2,d3,d4,d5,d6);
      elems.push_back(myElem);
    }
  }
  fprintf(stderr,"DONE with elements.\n");

  for(set<int>::iterator it=nodes.begin(); it!=nodes.end(); it++)
    fprintf(output,"%d %e %e %e\n", *it, X[*it-1][0], X[*it-1][1], X[*it-1][2]);
  fprintf(output,"%s %s %s %s\n", non1, "Local", non3, "LocalNodes");
  for(list<Elem>::iterator it=elems.begin(); it!=elems.end(); it++)
    fprintf(output,"%d %d %d %d %d %d\n", it->id, it->type, it->a, it->b, it->c, it->d);


  return 0;

}
