#include "FlExchange.h"
#include "../Comm/Communicator.h"
#include "../Math/Vector.h"
#include <cmath>

FlExchanger::FlExchanger(Communicator *com) : fluidCom(com) {
  sndParity = -1;
  rcvParity = -1;
  buffer = 0;
}

void
FlExchanger::getFluidLoad()
{

 int i,j,iDof;

 for(i=0; i<nbrReceivingFromMe; i++) {
     int fromNd;
     int tag =  FLTOSTMT + ((rcvParity > 0) ? 1 : 0) ;
     //_FORTRAN(nrecoc)(zero, tag, buffer, bufferLen, rsize, fromNd, toFluid);
     RecInfo rInfo = fluidCom->recFrom(tag, buffer, bufferLen);
     fromNd = rInfo.cpu;
     int origin = consOrigin[fromNd];
 }

 flipRcvParity();
}

//----------------------------------------------------------------------

void
FlExchanger::sendDisplacements(const SVec<double,3> &disp, int tag)  {


 int i,j;

 int pos = 0;
 for(i = 0; i < buffLen; ++i)
   buffer[i] = 0.0;
 for(i=0; i < nbrReceivingFromMe; i++) {

   int origPos = pos;
   for(j=0; j < nbSendTo[i]; ++j) {

     for(int k = 0; k < 3; ++k) {
       buffer[pos+k] = disp[ndNum[i][j]][k];
       buffer[pos+k+3] = 0.0;
     }
     pos += 6;

   }

   // int tag = STTOFLMT+( (sndParity > 0) ? 1 : 0);
   if(tag < 0) tag = STTOFLMT+( (sndParity > 0) ? 1 : 0);
   int fluidNode  = idSendTo[i];



   //_FORTRAN(nsedoc)(zero, tag, buffer, pos, fluidNode, toFluid);
   fluidCom->sendTo(fluidNode, tag, buffer+origPos, pos-origPos);

 }
 fluidCom->waitForAllReq();

 flipSndParity();

}

void
FlExchanger::sendParam(int algnum, double step, double totaltime,
                       int rstinc, int _isCollocated, double _a[2], int intParameter)
{
  int TNd  = 0;
  int thisNode = 0;

  double buffer[5];
  buffer[0] = (double) algnum;
  buffer[1] = (algnum==5)? step/2 : step;
  buffer[2] = totaltime;
  buffer[3] = (double) rstinc;
  buffer[4] = (double) intParameter; // Used to be AeroScheme now always conservative
  int msglen = 5;
  int tag = 3000;

  if(thisNode == 0) {
     fluidCom->sendTo(TNd, tag, buffer, msglen);
     fluidCom->waitForAllReq();
  }

}

void
FlExchanger::sendTempParam(int algnum, double step, double totaltime,
                       int rstinc)
{
  int TNd  = 0;
  int thisNode;
  thisNode = 0;

  double buffer[5];
  buffer[0] = (double) algnum;
  buffer[1] = step;
  buffer[2] = totaltime;
  buffer[3] = (double) rstinc;
  buffer[4] = (double) 2; // Used to be AeroScheme now always conservative
  int msglen = 5;
  int tag = 3000;

// Send to SUBROUTINE GETTEMPALG of Fluid Code
  if(thisNode == 0){
   fluidCom->sendTo(TNd, tag, buffer, msglen);
   fluidCom->waitForAllReq();
   }

}

// Temperature routines

void
FlExchanger::sendTemperature( const Vec<double> &T )
{
/*  nbrReceivingFromMe = number of fluid subdomains to which a structural
                         subdomain has to send values to.
    nbSendTo[i] = number of fluid nodes of fluid subdomain i communicating
                  to a structural subdomain
    buffer[0]   = fluid subdomain number
    buffer[1]   = number of information contained in one node  */

 int mynode = 0;

 // Prediction


 int i, j;
 for(i=0; i < nbrReceivingFromMe; i++) {

   int pos = 0;

   for(j=0; j < nbSendTo[i]; ++j) {

     buffer[pos] =T[ndNum[i][j]];

     pos += 1;
   }

   int tag = STTOFLHEAT;
   int fluidNode  = idSendTo[i];

// fprintf(stderr," STRUCT : Ready to send to Node %4d\n", tag);
// for (int xyz=0; xyz < pos; xyz++)
//      printf("Sending %4d = %14.7e\n",xyz,buffer[xyz]);

   //_FORTRAN(nsedoc)(zero, tag, buffer, pos, fluidNode, toFluid);
  fluidCom->sendTo(fluidNode, tag, buffer, pos);

//  fprintf(stderr," STRUCT : Done sending to Node %4d, Buffer %d\n", tag,pos);
//  fflush(stderr);

 }
 fluidCom->waitForAllReq();
}

void
FlExchanger::getFluidFlux(Vec<double> &flux)
{
/* sndTable's extensions dofs, xy, elemNum are declared in FlExchange.h
 in structure InterpPoint.
 nbrReceivingFromMe =  number of fluid subdomains */


  flux = 0.0;
 int i, j;

 for(i=0; i<nbrReceivingFromMe; i++) {
     int fromNd;
     int tag =  FLTOSTHEAT ;
     int rsize;
     RecInfo rInfo = fluidCom->recFrom(tag, buffer, bufferLen);
     fromNd = rInfo.cpu;
     rsize = rInfo.len;

     int origin = consOrigin[fromNd];

// Loop Over wet points of each fluid subdomain

     for(j=0; j<nbSendTo[origin]; ++j) {
       flux[ndNum[origin][j]] += buffer[j];
     }

 }

}

// This routine negotiate with the fluid codes which match points go where
void
FlExchanger::negotiate()
{

  int totSize = 0;
  int numFl = 0;
  // _FORTRAN(hetsize)(toFluid, numFl);
  numFl = fluidCom->remoteSize();
  int iFluid;
  int *flSize = new int[numFl];
  totmatch = 0;
  maxNode = -1;
  for(iFluid = 0; iFluid < numFl; ++iFluid) {
  fflush(stderr);
    int tag = FL_NEGOT;
    // double nFlMatched;
    int nFlMatched;
    int bufferLen = 1;
    int rsize, fromNd;
    RecInfo rInfo = fluidCom->recFrom(tag, &nFlMatched, bufferLen);
    fromNd = rInfo.cpu;
    flSize[fromNd] = nFlMatched;
    totmatch += nFlMatched;
  }

  //PHG
  if (totmatch == 0) {
    fprintf(stderr, " *** WARNING: by-passing negotiate step\n");
    fflush(stderr);
    return;
  }

  consOrigin = new int[numFl];
  nbSendTo = new int[numFl]; // should be actual number of senders
  idSendTo = new int[numFl]; // should be actual number of senders
  ndNum = new int*[numFl];
  int nSender = 0;
  for(iFluid = 0; iFluid < numFl; ++iFluid) {
    if(flSize[iFluid] > 0) {
      idSendTo[nSender] = iFluid;
      nbSendTo[nSender] = flSize[iFluid];
      consOrigin[iFluid] = nSender;
      ndNum[iFluid] = new int[flSize[iFluid]];
      totSize += flSize[iFluid];
      nSender++;
    }
  }
  nbrReceivingFromMe = nSender;
  //double *index = new double[totSize];
  int *index = new int[totSize];
  //InterpPoint **allPt = sndTable;

  //sndTable = new InterpPoint *[nSender];
  for(iFluid = 0; iFluid < nSender; ++iFluid) {
    int tag = FL_NEGOT+1;
    int bufferLen = totSize;
    int rsize, fromNd;
    //_FORTRAN(nrecoc)(zero, tag, index, bufferLen, rsize, fromNd, toFluid);
    RecInfo rInfo = fluidCom->recFrom(tag, index, bufferLen);
    fromNd = rInfo.cpu;
    rsize = rInfo.len;
    int sender = consOrigin[fromNd];
    for(int ipt = 0; ipt < nbSendTo[sender]; ++ipt) {
      ndNum[sender][ipt] = index[ipt];
      maxNode = std::max(maxNode, index[ipt]);
      index[ipt]=ipt; // to tell the fluid code we have this node
    }
    tag = FL_NEGOT;
    //double num = nbSendTo[sender];
    int num = nbSendTo[sender];
    int one = 1;
    //_FORTRAN(nsedoc)(zero, tag, &num, one, fromNd, toFluid);
    fluidCom->sendTo(fromNd, tag, &num, one);
    tag = FL_NEGOT+1;
    //_FORTRAN(nsedoc)(zero, tag, index, nbSendTo[sender], fromNd, toFluid);
    fluidCom->sendTo(fromNd, tag, index, nbSendTo[sender]);
    // To make sure we can reuse the buffers
    fluidCom->waitForAllReq();

  }

 delete [] index;

 buffer = new double[6*totSize];
 bufferLen = 6*totSize;
}
