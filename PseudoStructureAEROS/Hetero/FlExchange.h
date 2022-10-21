#ifndef _FLEXCHANGE_H_
#define _FLEXCHANGE_H_

class Communicator;
template <class S> class Vec;
template <class S, int n> class SVec;

class FlExchanger {
     double *buffer;
     double *buff;
     int bufferLen;
     int buffLen;

     int nbrReceivingFromMe;
     int *idSendTo;
     int *nbSendTo;
     int *consOrigin; // reverse table of idSendTo
     int **ndNum;
     int rcvParity, sndParity;

     Communicator *fluidCom;

     bool verboseFlag;
     int totmatch, maxNode;

   public:
     FlExchanger(Communicator *com);
     void negotiate();
     void sendDisplacements(const SVec<double,3> &, int tag=-1);
     void sendTemperature(const Vec<double> &);
     void getFluidLoad();
     void getFluidFlux(Vec<double> &flux);
     void sendParam(int alg, double step, double totalTime,
                    int restartinc, int _isCollocated, double alphas[2],
                    int intParam = 2);

     void sendTempParam(int algnum, double step, double totaltime,
                        int rstinc);
      void initSndParity(int pinit) { sndParity = pinit; }
      void initRcvParity(int pinit) { rcvParity = pinit; }
      void flipSndParity() { if(sndParity >= 0) sndParity = 1-sndParity; }
      void flipRcvParity() { if(rcvParity >= 0) rcvParity = 1-rcvParity; }
      int totNumDofs() { return totmatch; }
      int numNodes() { return maxNode+1; }
};

#define FLTOSTMT 1000
#define STTOFLMT 2000
#define STTOSTMT 4000
#define FLTOSTHEAT 5000
#define STTOFLHEAT 6000
#define STCMDMSG 8000
#define FLCMDMSG 9000
#define FL_NEGOT 10000

#endif
