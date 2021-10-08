#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
  int nGNodes = atoi(argv[1]);
  int nGElems = atoi(argv[2]);
  int List[100];

  double *X;
  FILE* source = 0, *output = 0;
  char non1[30], non2[30], non3[30], non4[30], non5[30]; 
  
  X = new double[nGNodes*3];
  int Id;

// load mesh.
  source = fopen("localMesh.top","r");
  output = fopen("localMesh2.top","w");
  if (!output) fprintf(stderr,"ERROR.\n");
  fscanf(source,"%s %s\n", non1, non2);
  fprintf(output,"%s %s\n", non1, non2);
  for (int i=0; i<nGNodes; i++) 
    fscanf(source,"%d %lf %lf %lf\n", &Id, &(X[3*i+0]), &(X[3*i+1]), &(X[3*i+2]));
  fprintf(stderr,"DONE with nodes.\n");

  fscanf(source,"%s %s %s %s\n", non1, non2, non3, non4);
  int d1[nGElems], d2[nGElems], d3[nGElems], d4[nGElems], d5[nGElems], d6[nGElems];
  int here = 0;
  for (int i=0; i<nGElems; i++) {
    fscanf(source, "%d %d %d %d %d %d\n", &(d1[i]), &(d2[i]), &(d3[i]), &(d4[i]), &(d5[i]), &(d6[i]));
    int found = 0;
    for (int j=0; j<here; j++) if (List[j]==d3[i]) {found = 1; break;}
    if (!found) List[here++] = d3[i];
      
    found = 0;
    for (int j=0; j<here; j++) if (List[j]==d4[i]) {found = 1; break;}
    if (!found) List[here++] = d4[i];
  
    found = 0;
    for (int j=0; j<here; j++) if (List[j]==d5[i]) {found = 1; break;}
    if (!found) List[here++] = d5[i];

    found = 0;
    for (int j=0; j<here; j++) if (List[j]==d6[i]) {found = 1; break;}
    if (!found) List[here++] = d6[i];
  }
  for (int i=0; i<here; i++) fprintf(output, "%d %e %e %e\n", List[i], X[3*(List[i]-1)], X[3*(List[i]-1)+1], X[3*(List[i]-1)+2]);
  fprintf(output,"%s %s %s %s\n", non1, non2, non3, non4);
  for (int i=0; i<nGElems; i++) {
    fprintf(output, "%d %d ", i+1, 5);
    for (int j=0; j<here; j++) if (List[j]==d3[i]) fprintf(output, "%d ", j+1);
    for (int j=0; j<here; j++) if (List[j]==d4[i]) fprintf(output, "%d ", j+1);
    for (int j=0; j<here; j++) if (List[j]==d5[i]) fprintf(output, "%d ", j+1);
    for (int j=0; j<here; j++) if (List[j]==d6[i]) fprintf(output, "%d\n", j+1);
  }


  if (X) delete[] X;
  return 0;

}
