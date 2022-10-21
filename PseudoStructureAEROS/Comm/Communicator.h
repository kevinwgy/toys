#ifndef _COMMUNICATOR_H_
#define _COMMUNICATOR_H_

//#include <Utils/MyComplex.h>
//class Connectivity;

#ifdef USE_MPI
#include <stdio.h>
#ifndef MPI_NO_CPPBIND
#define MPI_NO_CPPBIND
#endif
#include <mpi.h>
#include "../Utils/ResizeArray.h"

template <class T>
class CommTrace {
  public:
    static MPI_Datatype MPIType;
    static int multiplicity;
};

template <>
class CommTrace<bool> {
  private:
    static MPI_Datatype MPIType;
};

#endif

struct RecInfo {
  int cpu, len;
};


// Utility class to contain a message link information.
struct CPair {
  int locFrom, locTo;
  int glFrom, glTo;

};

int CPairCompare(void *a, void *b);

namespace Communication {
  template <typename Scalar>
    class Window;
}

class Communicator {

#ifdef USE_MPI
    MPI_Comm comm;
    int nPendReq;
    ResizeArray<MPI_Request> pendReq;
    ResizeArray<MPI_Status> reqStatus;
#endif

    int glNumCPU;
  public:
#ifdef USE_MPI
    Communicator(MPI_Comm, FILE * = stderr);
#endif
    Communicator(int _ncpu);
    Communicator(const Communicator &);
    template <class Type>
       Type globalSum(Type);
    bool globalMax(bool);
    template <class Type>
       Type globalMax(Type);
    template <class Type>
       Type globalMin(Type);
    template <class Type>
       void globalSum(int, Type*);
    template <class Type>
       void globalMax(int, Type*);
    template <class Type>
       void globalMin(int, Type*);
    template <class Type>
       void sendTo(int cpu, int tag, Type *data, int len);
    template <class Type>
       RecInfo recFrom(int tag, Type *data, int len);

    template <class Type>
       void allGather(Type *send_data, int send_count, Type *recv_data,
                      int recv_count);
    template <class Type>
       void allGatherv(Type *send_data, int send_count, Type *recv_data,
                      int recv_counts[], int displacements[]);

#ifdef USE_MPI
     template <class Type>
       void reduce(int num, Type*data, int root = 0, MPI_Op = MPI_SUM);
#endif

     template <class Type>
       void broadcast(int num, Type* data, int root = 0);

    void sync();
    int myID();
    int numCPUs();

#ifdef USE_MPI
    MPI_Comm* getCommunicator() { return &comm; }
#endif

  void split(int, int, Communicator**);
  void fatal(int res);
  int remoteSize();
  void waitForAllReq();

  template <typename Scalar>
    friend class Communication::Window;

  Communicator *merge(bool high); //<! returns the intra-communicator for the two groups of an inter-communicator
};

namespace Communication {

  template<typename Scalar>
  class Window {
#ifdef USE_MPI
    MPI_Win win;
#endif
    Communicator &com;
    Scalar *data;
  public:
    static const int Add=0, Min=1, Max=2;
    Window(Communicator &c, int size, Scalar *s);
    ~Window();
    void get(int locOff, int size, int prNum, int remOff);
    void put(Scalar *s, int locOff, int size, int prNum, int remOff);
    void accumulate(Scalar *s, int locOff, int size, int prNum, int remOff, int op);
    void fence(bool startOrEnd);
  };

}
/** allocate memory that can be used for one-sided communication */
 void* operator new(size_t, Communicator &c);
 void operator delete(void *p, Communicator &c);

 void *operator new[](size_t size, Communicator &c);

 void operator delete[](void *p, Communicator &c);

class SysCom : public Communicator {
    bool bootCom; // Whether this SysCom called MPI_Init
#ifdef USE_MPI
    MPI_Comm *salinasCommunicator;
#endif
  public:
    // SysCom(int argc, char *argv[]);
    SysCom(int &argc, char **&argv);
    SysCom();
    ~SysCom();
#ifdef USE_MPI
    SysCom(MPI_Comm *communicator);
    MPI_Comm* getCommunicator() { return salinasCommunicator; }
#endif
};

// The next routine provides tag from 100 to 200 cyclicaly
int uniqueTag();

extern SysCom *syscom;
extern Communicator *structCom;


#include <stdlib.h>
//#include <Utils/linkfc.h>
//#include <Utils/dbg_alloca.h>
#include <stdio.h>
#define _MAX_ALLOCA_SIZE 4096

template <class Type>
Type
Communicator::globalSum(Type data)
{
#ifdef USE_MPI
  Type buff;
  MPI_Allreduce(&data, &buff, 1, CommTrace<Type>::MPIType, MPI_SUM, comm);
  return buff;
#endif
}

template <class Type>
Type
Communicator::globalMax(Type data)
{
#ifdef USE_MPI
  Type buff;
  MPI_Allreduce(&data, &buff, 1, CommTrace<Type>::MPIType, MPI_MAX, comm);
  return buff;
#endif
}

template <class Type>
Type Communicator::globalMin(Type data)
{
#ifdef USE_MPI
  Type buff;
  MPI_Allreduce(&data, &buff, 1, CommTrace<Type>::MPIType, MPI_MIN, comm);
  return buff;
#endif
}

template <class Type>
void
Communicator::globalSum(int num, Type*data)
{
#ifdef USE_MPI
  Type *work;
  //dbg_alloca(0);

  //int segSize = (num > 65536) ? 65536 : num;
  int segSize = (num > 4096) ? 4096 : num; // PJSA 6-19-07

  if(segSize > 5000)
    work = new Type[segSize];
  else
    work = (Type *)alloca(segSize*sizeof(Type));

  int offset;
  for(offset = 0; offset < num; offset +=segSize) {
    int msgSize = (num-offset < segSize) ? num-offset : segSize;
    MPI_Allreduce(data+offset, work, msgSize,
                  CommTrace<Type>::MPIType, MPI_SUM, comm);
    for(int i = 0; i < msgSize; ++i)
      data[offset+i] = work[i];
  }
  if(segSize > 5000)
    delete [] work;
#endif
}

template <class Type>
void
Communicator::globalMax(int num, Type*data)
{
#ifdef USE_MPI
  Type *work;
  //dbg_alloca(0);

  //int segSize = (num > 65536) ? 65536 : num;
  int segSize = (num > 4096) ? 4096 : num; // PJSA 6-19-07

  if(segSize > 5000)
    work = new Type[segSize];
  else
    work = (Type *)alloca(segSize*sizeof(Type));

  int offset;
  for(offset = 0; offset < num; offset +=segSize) {
    int msgSize = (num-offset < segSize) ? num-offset : segSize;
    MPI_Allreduce(data+offset, work, msgSize,
                  CommTrace<Type>::MPIType, MPI_MAX, comm);
    for(int i = 0; i < msgSize; ++i)
      data[offset+i] = work[i];
  }
  if(segSize > 5000)
    delete [] work;
#endif
}

template <class Type>
void
Communicator::globalMin(int num, Type*data)
{
#ifdef USE_MPI
  Type *work;
  //dbg_alloca(0);

  //int segSize = (num > 65536) ? 65536 : num;
  int segSize = (num > 4096) ? 4096 : num; // PJSA 6-19-07

  if(segSize > 5000)
    work = new Type[segSize];
  else
    work = (Type *)alloca(segSize*sizeof(Type));

  int offset;
  for(offset = 0; offset < num; offset +=segSize) {
    int msgSize = (num-offset < segSize) ? num-offset : segSize;
    MPI_Allreduce(data+offset, work, msgSize,
                  CommTrace<Type>::MPIType, MPI_MIN, comm);
    for(int i = 0; i < msgSize; ++i)
      data[offset+i] = work[i];
  }
  if(segSize > 5000)
    delete [] work;
#endif
}

template <class Type>
void
Communicator::sendTo(int cpu, int tag, Type *buffer, int len)
{
#ifdef USE_MPI
  int thisReq = nPendReq++;
  MPI_Request *req = pendReq+thisReq;
  MPI_Isend(buffer, len, CommTrace<Type>::MPIType,
            cpu, tag, comm, req);
#endif
}

template <class Type>
RecInfo
Communicator::recFrom(int tag, Type *buffer, int len)
{
#ifdef USE_MPI
  RecInfo rInfo;
  MPI_Status status;
  MPI_Recv(buffer, len,
           CommTrace<Type>::MPIType, MPI_ANY_SOURCE, tag, comm, &status);
  MPI_Get_count(&status, CommTrace<Type>::MPIType, &rInfo.len);
  rInfo.cpu = status.MPI_SOURCE;
  return rInfo;
#endif
}

template <class Type>
void
Communicator::allGather(Type *send_data, int send_count,
                        Type *recv_data, int recv_count)
{
#ifdef USE_MPI
  MPI_Allgather(send_data, send_count, CommTrace<Type>::MPIType,
                recv_data, recv_count, CommTrace<Type>::MPIType, comm);
#endif
}

template <class Type>
void
Communicator::allGatherv(Type *send_data, int send_count,
                         Type *recv_data, int recv_counts[], int displacements[])
{
#ifdef USE_MPI
  MPI_Allgatherv(send_data, send_count, CommTrace<Type>::MPIType,
                 recv_data, recv_counts, displacements,
                 CommTrace<Type>::MPIType, comm);
#endif
}

#define _MESSAGE_SIZE 100000
#ifdef USE_MPI
template <class Type>
void
Communicator::reduce(int num, Type* data, int root, MPI_Op mpi_op)
{
  int maxSegSize = _MESSAGE_SIZE/sizeof(Type);
  int segSize = (num > maxSegSize) ? maxSegSize : num;
  Type *buffer;

  if(segSize > _MAX_ALLOCA_SIZE)
    buffer = new Type[segSize];
  else {
    alloca(0);
    buffer = (Type *)alloca(segSize*sizeof(Type));
  }

  for(int offset = 0; offset < num; offset +=segSize) {
    int count = (num-offset < segSize) ? num-offset : segSize;
    MPI_Reduce(data+offset, buffer, count, CommTrace<Type>::MPIType, /*MPI_SUM*/ mpi_op, root, comm);
    for(int i = 0; i < count; ++i) data[offset+i] = buffer[i];
  }
  if(segSize > _MAX_ALLOCA_SIZE)
    delete [] buffer;
}
#endif

template <class Type>
void
Communicator::broadcast(int num, Type* data, int root)
{
#ifdef USE_MPI
  int maxSegSize = _MESSAGE_SIZE/sizeof(Type);
  int segSize = (num > maxSegSize) ? maxSegSize : num;
  Type *buffer;

  if(segSize > _MAX_ALLOCA_SIZE)
    buffer = new Type[segSize];
  else {
    alloca(0);
    buffer = (Type *)alloca(segSize*sizeof(Type));
  }

  for(int offset = 0; offset < num; offset +=segSize) {
    int count = (num-offset < segSize) ? num-offset : segSize;
    if(myID() == root) for(int i = 0; i < count; i++) buffer[i] = data[offset+i];
    MPI_Bcast(buffer, count, CommTrace<Type>::MPIType, root, comm);
    if(myID() != root) for(int i = 0; i < count; ++i) data[offset+i] = buffer[i];
  }
  if(segSize > _MAX_ALLOCA_SIZE)
    delete [] buffer;
#endif
}


#ifdef NO_COMPLEX_MPI
// PJSA 1-7-2008 specializations of communication functions for platforms which do not support MPI_COMPLEX_DOUBLE
// implemented in Driver.d/MPIComm.C
template <>
complex<double> Communicator::globalSum(complex<double> data);

template <>
complex<double> Communicator::globalMax(complex<double> data);

template <>
complex<double> Communicator::globalMin(complex<double> data);

template <>
void Communicator::globalSum(int num, complex<double>* data);

template <>
void Communicator::sendTo(int cpu, int tag, complex<double> *buffer, int len);

template <>
RecInfo Communicator::recFrom(int tag, complex<double> *buffer, int len);

template <>
void Communicator::allGather(complex<double> *send_data, int send_count, complex<double> *recv_data, int recv_count);

template <>
void Communicator::allGatherv(complex<double> *send_data, int send_count, complex<double> *recv_data, int recv_counts[], int displacements[]);
#endif
namespace Communication {

  template <typename Scalar>
  Window<Scalar>::Window(Communicator &c, int size, Scalar *s) : com(c) {
#ifdef USE_MPI
    MPI_Win_create(s, size*sizeof(Scalar), sizeof(Scalar), MPI_INFO_NULL,
        com.comm, &win);
#endif
  }
  template <typename Scalar>
    Window<Scalar>::~Window() {
  #ifdef USE_MPI
      MPI_Win_free(&win);
  #endif
    }

  template <typename Scalar>
  void Window<Scalar>::accumulate(Scalar *a, int locOff, int size, int prNum, int remOff, int op) {
#ifdef USE_MPI
    static const MPI_Op mpiOp[] = { MPI_SUM, MPI_MIN, MPI_MAX };
    MPI_Accumulate(a+locOff, size*CommTrace<Scalar>::multiplicity, CommTrace<Scalar>::MPIType,
        prNum,
        remOff*CommTrace<Scalar>::multiplicity, size*CommTrace<Scalar>::multiplicity,
        CommTrace<Scalar>::MPIType, mpiOp[op],
        win);
#endif
      }

  template <typename Scalar>
  void Window<Scalar>::put(Scalar *a, int locOff, int size, int prNum, int remOff) {
#ifdef USE_MPI
    static const MPI_Op mpiOp[] = { MPI_SUM, MPI_MIN, MPI_MAX };
    MPI_Put(a+locOff, size*CommTrace<Scalar>::multiplicity, CommTrace<Scalar>::MPIType,
        prNum,
        remOff*CommTrace<Scalar>::multiplicity, size*CommTrace<Scalar>::multiplicity,
        CommTrace<Scalar>::MPIType,
        win);
#endif
      }

  template <typename Scalar>
    void Window<Scalar>::fence(bool isBeginning) {
      if(isBeginning)
        MPI_Win_fence((MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE), win);
      else
        MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
  }
}
#endif
