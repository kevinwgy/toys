#include <cstring>
#include <iostream>
#include <fstream>
#include "Hetero/FlExchange.h"
#include "Comm/Communicator.h"
#include "Math/Vector.h"

#define MAX_CODES 4
#define FLUID_ID 0
#define STRUC_ID 1
#define HEAT_ID  2
#define EMBED_ID 3

Communicator *worldCom = 0;
int nNodes, mode;
double tMax, dt, omega, dx, dy, dz;

void parseArgs(int argc, char **argv) {
  if(argc < 9) {
    fprintf(stderr, "Error:\nUsage: %s <number of nodes> <1 or 0> <dt> <tMax> <omega> <dx> <dy> <dz>\n", argv[0]);
    worldCom->fatal(-1);
    exit(-1);
  }

  nNodes = atoi(argv[1]);
  mode = atoi(argv[2]);
  dt = atof(argv[3]);
  tMax = atof(argv[4]);
  omega = atof(argv[5]);
  dx = atof(argv[6]);
  dy = atof(argv[7]);
  dz = atof(argv[8]);
}


void getDisp(SVec<double,3> &disp)
{



}

void doTheWork(int argc, char **argv) {
  parseArgs(argc, argv);
  std::cout << "Number of nodes " << nNodes
            << std::endl;
  Communicator *allCom[MAX_CODES];
  int myId = mode==0 ? EMBED_ID : FLUID_ID;
  int otherId = mode!=0 ? EMBED_ID : FLUID_ID;
  worldCom->split(myId, MAX_CODES, allCom);
  Communicator *flComMech = allCom[otherId];
  Communicator *intracom = flComMech->merge(mode != 0);
  std::cout << myId << " " << otherId << " Other com is: " << flComMech << std::endl;
  std::cout << "And I am at " << intracom->myID() << std::endl;
  double (*data)[3] = new (*intracom) double[nNodes][3];
  SVec<double, 3> f(nNodes, data);
  bool firstIt = true;

  //----------- load the structure surface (debug) -----------------
  FILE *nodeFile = 0;
  nodeFile = fopen("corrected_solidSurface.top","r");
  if(!nodeFile) fprintf(stderr,"top file not found (in structure codes)\n");
  int nNodes2, nElems;
  fscanf(nodeFile, "%d %d\n", &nNodes2, &nElems);
  double X[nNodes2][3];
  int nothing;
  for (int iNode=0; iNode<nNodes2; iNode++) 
    fscanf(nodeFile,"%d %lf %lf %lf\n", &nothing, &(X[iNode][0]), &(X[iNode][1]), &(X[iNode][2]));
  fclose(nodeFile);
  //----------------------------------------------------------------


   //send timestep.
  {
    Communication::Window<double> win0(*intracom, 1, &dt);
    win0.fence(true);
    for(int i = 1; i < intracom->numCPUs(); ++i) {
      std::cout << "Sending the timestep (" << dt << ") to fluid " << i << std::endl;
      win0.put(&dt, 0, 1, i, 0);
    }
    win0.fence(false);
  }

  for(double t = 0; t < tMax-0.01*dt; t += dt) {

    //receive force. (not at the initialization stage)
    if(mode == 0 && !firstIt) {
      Communication::Window<double> win(*intracom, 3*nNodes, (double *)data);
      f = 0.0;
      win.fence(true);
      win.fence(false);
      std::cout << intracom->myID() << " : ";
      double fx=0, fy=0, fz=0; //write the total froce. 
      for(int i = 0; i < nNodes; ++i) {
	fx += f[i][0];
	fy += f[i][1];
	fz += f[i][2];
      }
      std::cout << "Total force: " << fx << " " << fy << " " << fz << std::endl;
    } else
      firstIt = false;

    //send displacement.
    if(mode == 0) {
      for(int i = 0; i < nNodes; ++i) {
	f[i][0] = (1-cos(omega*(t+dt)))*dx;
	f[i][1] = (1-cos(omega*(t+dt)))*dy;
	f[i][2] = (X[i][1]*X[i][1])*(1-cos(omega*(t+dt)))*dz;
      }
      Communication::Window<double> win2(*intracom, 3*nNodes, (double *)data);
      win2.fence(true);
      std::cout << "Sending a displacement to fluid ";
      for(int i = 1; i < intracom->numCPUs(); ++i) {
        std::cout << i;
	win2.put((double *)data, 0, 3*nNodes, i, 0);
        if(i!=intracom->numCPUs()-1) std::cout << ", ";
      }
      std::cout << std::endl;
      win2.fence(false);
    } else{
      Communication::Window<double> win(*intracom, 3*nNodes, (double *)data);
      for(int i = 0; i < nNodes; ++i) {
        f[i][0] = 10*i;
        f[i][1] = -10*i;
        f[i][2] = i;
      }
      win.fence(true);
      win.accumulate((double *)data, 0, 3*nNodes, 0, 0, Communication::Window<double>::Add);
      win.fence(false);
      win.fence(true);
      win.fence(false);
    }
  }

}

int main(int argc, char **argv)
{
  SysCom theCom(argc, argv);
  worldCom = &theCom;
  doTheWork(argc, argv);
  return 0;
}

extern "C" int entrypoint(int argc, char **argv)
{
  SysCom theCom;
  worldCom = &theCom;
  doTheWork(argc, argv);
  return 0;
}

