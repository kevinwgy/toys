#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <list>
#include <set>
#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{
  double xyz_min[3] = {0.0, 0.0, 0.0};
  double xyz_max[3] = {12000.0, 10000.0, 10000.0};
  
  double dxyz_min[3] = {50.0, 50.0, 50.0};
  double dxyz_max[3] = {800.0, 800.0, 800.0};
  double c2xyz[3]    = {0.1, 0.1, 0.1};
  double dxyz_breaker[3] = {800.0, 1200.0, 1200.0};
  for (int i=0; i<3; i++)
    c2xyz[i] *= xyz_max[i] - dxyz_breaker[i];

  int N1Dmax = 10000;
  int N[3] = {0,0,0};

  FILE* output = fopen("mesh.include","w");
  // between breaker and xyz_max, the mesh size varies as 
  // dx = 1.0/( (1/dxmin-1/dxmax)*exp(-(x-breaker)/2c2) + 1/dxmax )

  double **xyz = new double*[3];
  for (int i=0; i<3; i++)
    xyz[i] = new double[N1Dmax];

  // calc. grid points
  for (int iDim=0; iDim<3; iDim++) {
    double xhere = xyz_min[iDim];
    double dx = dxyz_min[iDim];
    xyz[iDim][N[iDim]++] = xhere;
    while (1) {
      if (xhere>xyz_max[iDim])
        break;
      if (xhere>dxyz_breaker[iDim]) {
        dx = 1.0/(  (1.0/dxyz_min[iDim]-1.0/dxyz_max[iDim])*
                    exp(-(xhere-dxyz_breaker[iDim])/(2.0*c2xyz[iDim]))
                  + 1.0/dxyz_max[iDim] );
//        cout << "exp = " << exp(-(xhere-dxyz_breaker[iDim])/(2.0*c2xyz[iDim])) << endl;
//        cout << "xhere = " << xhere << ", dx = " << dx << "." << endl;
      } else
        dx = dxyz_min[iDim];
      xhere += dx;
      xyz[iDim][N[iDim]++] = xhere;
    }
//    cout << "Dim " << iDim+1 << ": " << N[iDim] << " nodes, max = " << xyz[iDim][N[iDim]-1]
//         << "." << endl; 
  }

  // write nodes 
  int iNode = 0;
  fprintf(output,"NODES\n");
  for (int i=0; i<N[0]; i++)
    for (int j=0; j<N[1]; j++)
      for (int k=0; k<N[2]; k++)
        fprintf(output,"%8d  %12.8e  %12.8e  %12.8e\n", ++iNode, xyz[0][i], xyz[1][j], xyz[2][k]);
  cout << "Done with NODES. nNode = " << iNode << "." << endl;

  // write elements
  int iElem = 0;
  fprintf(output,"TOPOLOGY\n");
  for (int i=0; i<N[0]-1; i++)
    for (int j=0; j<N[1]-1; j++)
      for (int k=0; k<N[2]-1; k++) {
        int n1 = i*(N[1]*N[2]) + j*N[2] + k + 1;
        int n2 = n1 + N[1]*N[2];
        int n3 = n2 + N[2];
        int n4 = n1 + N[2];
        int n5 = n1 + 1;
        int n6 = n2 + 1;
        int n7 = n3 + 1;
        int n8 = n4 + 1;
        fprintf(output,"%8d %4d %8d %8d %8d %8d %8d %8d %8d %8d\n", ++iElem, (int)17,
                n1, n2, n3, n4, n5, n6, n7, n8);
      }
  cout << "Done with TOPOLOGY. nElem = " << iElem << "." << endl;

  // write triangulated surface {y=ymin}
  iElem = 0;
  fprintf(output,"SURFACETOPO 5\n");
  for (int i=0; i<N[0]-1; i++)
    for (int k=0; k<N[2]-1; k++) {
      int n1 = i*(N[1]*N[2]) + k + 1;
      int n2 = n1 + N[1]*N[2];
      int n3 = n2 + 1;
      int n4 = n1 + 1;
      fprintf(output,"%8d %4d %8d %8d %8d\n", ++iElem, (int)3, n1, n2, n3);
      fprintf(output,"%8d %4d %8d %8d %8d\n", ++iElem, (int)3, n1, n3, n4);
    }
  cout << "Done with SURFACETOPO 5. nElem = " << iElem << "." << endl;

  // write triangulated surface {z=zmin}
  iElem = 0;
  fprintf(output,"SURFACETOPO 6\n");
  for (int i=0; i<N[0]-1; i++)
    for (int j=0; j<N[1]-1; j++) {
      int n1 = i*(N[1]*N[2]) + j*N[2] + 1;
      int n2 = n1 + N[1]*N[2];
      int n3 = n2 + N[2];
      int n4 = n1 + N[2];
      fprintf(output,"%8d %4d %8d %8d %8d\n", ++iElem, (int)3, n1, n4, n2); //"outward"
      fprintf(output,"%8d %4d %8d %8d %8d\n", ++iElem, (int)3, n2, n4, n3); //"outward"
    }
  cout << "Done with SURFACETOPO 6. nElem = " << iElem << "." << endl;

  // write triangulated surface {z=zmin}
  iElem = 0;
  fprintf(output,"SURFACETOPO 8\n");
  for (int j=0; j<N[1]-1; j++)
    for (int k=0; k<N[2]-1; k++) {
      int n1 = j*N[2] + k + 1;
      int n2 = n1 + 1;
      int n3 = n2 + N[2];
      int n4 = n3 - 1;
      fprintf(output,"%8d %4d %8d %8d %8d\n", ++iElem, (int)3, n1, n2, n3); //"outward"
      fprintf(output,"%8d %4d %8d %8d %8d\n", ++iElem, (int)3, n1, n3, n4); //"outward"
    }
  cout << "Done with SURFACETOPO 8. nElem = " << iElem << "." << endl;

  //clean up
  fclose(output);
  for (int i=0; i<3; i++)
    delete [] xyz[i];
  delete [] xyz;

  return 0;
}
