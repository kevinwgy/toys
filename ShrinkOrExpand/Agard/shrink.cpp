#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main (int argc, char* argv[]) 
{
  //FILE *input = fopen("/lustre/home/kevinwgy/Meshes/Agard/fine/agard_in_out.top","r");
  //FILE *input = fopen("/lustre/home/kevinwgy/Meshes/Agard/coarse/Agard_in_out.top","r");
  //FILE *input = fopen("/lustre/home/kevinwgy/Meshes/Agard/finer/agard-3.7M-in-out.top","r");
  FILE *input = fopen("/lustre/home/kevinwgy/Meshes/Agard/finest/agard-7.5M-in-out.top","r");
  FILE *output = fopen("shrunk_finest_Agard_in_out.top","w");
  char word1[20], word2[20], word3[20], word4[20], word5[20];
  int integer1, integer2, integer3, integer4, integer5, integer6;
  double x,y,z;

  double Ox = 11.0, Oy = 0.0, Oz = 0.0;
  //double cx = 1.02, cy = 1.003, cz = 1.08;
  double cx = 0.98, cy = 0.997, cz = 0.92;

  
//  int nNodes = 105030;
//  int nNodes = 1191248;
//  int nNodes = 3785354;
  int nNodes = 7536495;
  fscanf(input, "%s %s\n", word1, word2);
  fprintf(output, "%s %s\n", word1, word2);
  for (int i=0; i<nNodes; i++) {
    fscanf(input, "%d %lf %lf %lf\n", &integer1, &x, &y, &z);
    x = cx*(x - Ox) + Ox;
    y = cy*(y - Oy) + Oy + 0.003;
    z = cz*(z - Oz) + Oz;
    fprintf(output, "%d %e %e %e\n", integer1, x, y, z);
  }
  fprintf(stderr,"DONE with nodes.\n");
//  int nElems1 = 6971318;
//  int nElems1 = 609576;
//  int nElems1 = 22100098;
  int nElems1 = 43972986;
  fscanf(input, "%s %s %s %s\n", word1, word2, word3, word4);
  fprintf(output, "%s %s %s %s\n", word1, word2, word3, word4);
  for (int i=0; i<nElems1; i++) {
    fscanf(input, "%d %d %d %d %d %d\n", &integer1, &integer2, &integer3, &integer4, &integer5, &integer6);
    fprintf(output, "%d %d %d %d %d %d\n", integer1, integer2, integer3, integer4, integer5, integer6);
  }

//  int nElems2 = 703;
//  int nElems2 = 160;
//  int nElems2 = 705;
  int nElems2 = 1298;
  fscanf(input, "%s %s %s %s\n", word1, word2, word3, word4);
  fprintf(output, "%s %s %s %s\n", word1, word2, word3, word4);
  for (int i=0; i<nElems2; i++) {
    fscanf(input, "%d %d %d %d %d\n", &integer1, &integer2, &integer3, &integer4, &integer5);
    fprintf(output, "%d %d %d %d %d\n", integer1, integer2, integer3, integer4, integer5);
  }
  
//  int nElems3 = 5843;
//  int nElems3 = 2142;
//  int nElems3 = 9219;
  int nElems3 = 13656;
  fscanf(input, "%s %s %s %s\n", word1, word2, word3, word4);
  fprintf(output, "%s %s %s %s\n", word1, word2, word3, word4);
  for (int i=0; i<nElems3; i++) {
    fscanf(input, "%d %d %d %d %d\n", &integer1, &integer2, &integer3, &integer4, &integer5);
    fprintf(output, "%d %d %d %d %d\n", integer1, integer2, integer3, integer4, integer5);
  }
  fprintf(stderr,"DONE.\n");
  fclose(input);
  fclose(output);
  return 0;
}
