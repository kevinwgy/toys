#include <stdlib.h>
#include <iostream>

#include <stdio.h>
//#include <Utils.d/MyComplex.h>

#ifdef TFLOP
#include "mpi.h"
#define SEGMENT_GLBSUM
#endif

FILE *debugFile = 0;

#include "Communicator.h"

#ifndef MPI_INTEGER
#define MPI_INTEGER MPI_INT
#endif

#ifndef MPI_CHARACTER
#define MPI_CHARACTER MPI_CHAR
#endif

// PJSA: for linux
#if (defined(USE_MPI) && !defined(MPI_DOUBLE_COMPLEX) && defined(LAM_MPI))
//#include<mpisys.h>
#define MPI_DOUBLE_COMPLEX ((MPI_Datatype) &lam_mpi_cxx_dblcplex)
#endif
// CRW - this define is a hack to let it compile on sun, but feti-h will not work
#if (defined(USE_MPI) && !defined(MPI_DOUBLE_COMPLEX) && defined(NO_COMPLEX_MPI))    //CRW
#define MPI_DOUBLE_COMPLEX MPI_LONG_DOUBLE    //CRW
#endif    //CRW

MPI_Datatype xxx;
#ifdef CXX  // for DEC cxx compiler
MPI_Datatype CommTrace<double>::MPIType = MPI_DOUBLE;
MPI_Datatype CommTrace<float>::MPIType = MPI_FLOAT;
MPI_Datatype CommTrace<int>::MPIType = MPI_INTEGER;
MPI_Datatype CommTrace<char>::MPIType = MPI_CHARACTER;
//MPI_Datatype CommTrace<DComplex>::MPIType = MPI_DOUBLE_COMPLEX;
//MPI_Datatype CommTrace<bool>::MPIType = MPI_BOOL;
#else
template<>
MPI_Datatype CommTrace<double>::MPIType = MPI_DOUBLE;
template<>
MPI_Datatype CommTrace<float>::MPIType = MPI_FLOAT;
template<>
MPI_Datatype CommTrace<int>::MPIType = MPI_INTEGER;
template<>
MPI_Datatype CommTrace<char>::MPIType = MPI_CHARACTER;
template<>
int CommTrace<double>::multiplicity =  1;
//template<>
//MPI_Datatype CommTrace<DComplex>::MPIType = MPI_DOUBLE_COMPLEX;
//template<>
//MPI_Datatype CommTrace<bool>::MPIType = MPI_BOOL;
#endif

static MPI_Request nullReq;
static MPI_Status nullStat;

//------------------------------------------------------------------------------
// Communicator *fetiCom;

Communicator::Communicator(int _ncpu)
: nPendReq(0), pendReq(nullReq) , reqStatus(nullStat)
{
 comm = MPI_COMM_WORLD;
 nPendReq = 0;
 glNumCPU = _ncpu;
}

void
Communicator::sync()
{
 MPI_Barrier(comm);
}

int
Communicator::myID()
{
 int id;
 MPI_Comm_rank(comm, &id);
 return id;
}

SysCom::SysCom(int &argc, char **&argv) : Communicator(MPI_COMM_WORLD, stderr)
{
  salinasCommunicator=0;
#ifdef USE_MPI
  MPI_Init(&argc,&argv);
  bootCom = true;
#endif
}

#ifdef DISTRIBUTED
// XML Needs to be reexamined
SysCom::SysCom(MPI_Comm *mpi_comm) : Communicator(*mpi_comm,stderr)
{
 salinasCommunicator=mpi_comm;
 bootCom = false;

#ifdef TFLOP
  int structID = 2;
 _FORTRAN(startcom)(mpi_comm, structID);
#endif
}
#endif

// XML Needs to be reexamined
SysCom::SysCom() : Communicator(MPI_COMM_WORLD,stderr)
{
  salinasCommunicator=0;
  bootCom = false;
}

SysCom::~SysCom()
{
#ifdef USE_MPI
 if(bootCom) {
   MPI_Finalize();
 }
#endif
}

int
Communicator::numCPUs()
{
 int id;
  MPI_Comm_size(comm, &id);
 return id;
}

int CPairCompare(const void *a, const void *b)
{
 CPair *pa = (CPair *)a;
 CPair *pb = (CPair *)b;

 int mina, minb, maxa, maxb;
 if(pa->glFrom < pa->glTo) {
   mina = pa->glFrom;
   maxa =  pa->glTo;
 } else {
  mina = pa->glTo;
  maxa = pa->glFrom;
 }

 if(pb->glFrom < pb->glTo) {
   minb = pb->glFrom;
   maxb =  pb->glTo;
 } else {
  minb = pb->glTo;
  maxb = pb->glFrom;
 }

 if(mina < minb || (mina == minb && maxa < maxb)) return -1;
 if(mina == minb && maxa == maxb) return 0;
 return 1;
}

static int v = 0;

int uniqueTag()
{
 v = (v+1)%100;
 return v+100;
}

void Communicator::waitForAllReq()
{
  // Just making sure that reqStatus has an appropriate length
  MPI_Status *safe = reqStatus+nPendReq;

  if (safe == 0)
    exit(1);

  int nSuccess = MPI_Waitall(nPendReq, pendReq+0, reqStatus+0);

  if (nSuccess) {
    fprintf(stderr, " *** ERROR: unexpected success number %d\n", nSuccess);
    exit(1);
  }

  nPendReq = 0;
}

void Communicator::split(int color, int maxcolor, Communicator** c)
{
  int i;
  for (i=0; i<maxcolor; ++i)
    c[i] = 0;

#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm comm1;
  MPI_Comm_split(comm, color+1, rank, &comm1); //wrong should use color
  c[color] = new Communicator(comm1,stderr);

  int* leaders = new int[maxcolor];
  int* newleaders = new int[maxcolor];
  for (i=0; i<maxcolor; ++i)
    leaders[i] = -1;

  int localRank;
  MPI_Comm_rank(comm1, &localRank);
  if (localRank == 0)
    leaders[color] = rank;

  MPI_Allreduce(leaders, newleaders, maxcolor, MPI_INTEGER, MPI_MAX, comm);

   for (i=0; i<maxcolor; ++i) {
     if (i != color && newleaders[i] >= 0) {
       int tag;
       if (color < i)
        tag = maxcolor*(color+1)+i+1;
       else
        tag = maxcolor*(i+1)+color+1;
       MPI_Comm comm2;
       MPI_Intercomm_create(comm1, 0, comm, newleaders[i], tag, &comm2);
       c[i] = new Communicator(comm2,stderr);
     }
   }

  if (leaders)
    delete [] leaders;
  if (newleaders)
    delete [] newleaders;
#else
  c[color] = this;
#endif
}

Communicator *Communicator::merge(bool high) {
#ifdef USE_MPI
  MPI_Comm newComm;
  MPI_Intercomm_merge(comm, high ? 1 : 0, &newComm);
  return new Communicator(newComm, NULL);
#endif
  return 0;
}

#ifdef USE_MPI
Communicator::Communicator(MPI_Comm c1, FILE *fp)
  : pendReq(nullReq), reqStatus(nullStat)
{
  comm = c1;
  nPendReq = 0;
}
#endif

Communicator::Communicator(const Communicator &c1)
: nPendReq(0), pendReq(nullReq) , reqStatus(nullStat)
{
#ifdef USE_MPI
 comm = c1.comm;
 MPI_Comm_size(comm, &glNumCPU);
#endif
}

int
Communicator::remoteSize()
{
 int numRemote = 0;

#ifdef USE_MPI
  MPI_Comm_remote_size(comm, &numRemote);
#endif

  return numRemote;
}

bool Communicator::globalMax(bool b)
{
  char data = (b) ? 1 : 0;
  char buff;
 #ifdef USE_MPI

   MPI_Allreduce(&data, &buff, 1, CommTrace<char>::MPIType, MPI_MAX,
     comm);

 #endif
   return buff != 0;
}

void Communicator::fatal(int errCode) {
#ifdef USE_MPI
  MPI_Abort(comm, errCode);
#else
  exit(errCode);
#endif
}

#ifdef NO_COMPLEX_MPI
template <>
complex<double> Communicator::globalSum(complex<double> data)
{
  double tmp_data[2] = { data.real(), data.imag() };
  globalSum(2, tmp_data);
  return complex<double>(tmp_data[0],tmp_data[1]);
}

template <>
complex<double> Communicator::globalMax(complex<double> data)
{
  cerr << "ERROR: Communicator::globalMax called with complex data\n";
  return complex<double>(0.0,0.0);
}

template <>
complex<double> Communicator::globalMin(complex<double> data)
{
  cerr << "ERROR: Communicator::globalMin called with complex data\n";
  return complex<double>(0.0,0.0);
}

template <>
void Communicator::globalSum(int num, complex<double> *data)
{
  double *tmp_data = new double[2*num];
  for(int i=0; i<num; ++i) { tmp_data[2*i] = data[i].real(); tmp_data[2*i+1] = data[i].imag(); }
  globalSum(2*num,tmp_data);
  for(int i=0; i<num; ++i) data[i] = complex<double>(tmp_data[2*i],tmp_data[2*i+1]);
  delete [] tmp_data;
}

template <>
void Communicator::sendTo(int cpu, int tag, complex<double> *buffer, int len)
{
  cerr << "ERROR: Communicator::sendTo called with complex data\n";
}

template <>
RecInfo Communicator::recFrom(int tag, complex<double> *buffer, int len)
{
  cerr << "ERROR: Communicator::recFrom called with complex data\n";
  return RecInfo();
}

template <>
void Communicator::allGather(complex<double> *send_data, int send_count, complex<double> *recv_data, int recv_count)
{
  cerr << "ERROR: Communicator::allGather called with complex data\n";
}

template <>
void Communicator::allGatherv(complex<double> *send_data, int send_count, complex<double> *recv_data, int recv_counts[], int displacements[])
{
  cerr << "ERROR: Communicator::allGatherv called with complex data\n";
}
#endif

//------------------------------------------------------------------------------

void* operator new(size_t size, Communicator &) {
  void *a;
#ifdef USE_MPI
        MPI_Alloc_mem(size, MPI_INFO_NULL, &a);

#else // USE_MPI
        a = malloc(size);
#endif // USE_MPI
        if( !a ) {

            std::bad_alloc ba;

            throw ba;
          }
        return a;
}

void operator delete(void *p, Communicator &) {
#ifdef USE_MPI
        MPI_Free_mem(p);
#else // USE_MPI
        free(p);
#endif // USE_MPI
}

void* operator new[](size_t size, Communicator &) {
        void *a;
#ifdef USE_MPI
        MPI_Alloc_mem(size, MPI_INFO_NULL, &a);
#else // USE_MPI
        a = malloc(size);
#endif // USE_MPI
        if( !a ) {

            std::bad_alloc ba;

            throw ba;
          }
        return a;
}

void operator delete[](void *p, Communicator &) {
#ifdef USE_MPI
        MPI_Free_mem(p);
#else // USE_MPI
        free(p);
#endif // USE_MPI
}
