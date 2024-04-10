#include<mpi.h>
#include<vector>
#include<cassert>
using namespace std;

#define SOLVER3_COMM_TAG 2

int main(int argc, char *argv[])
{

  MPI_Init(NULL,NULL);

  int global_rank(0), global_size(0);
  MPI_Comm global_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(global_comm, &global_rank);
  MPI_Comm_size(global_comm, &global_size);

  MPI_Comm comm;
  MPI_Comm_split(global_comm, SOLVER3_COMM_TAG+1, global_rank, &comm);

  int rank(0), size(0);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  fprintf(stdout,"I am proc %d (of %d) --- global %d (of %d): running solver 3.\n",
          rank, size, global_rank, global_size);

  int maxcolor = 10;
  int color = SOLVER3_COMM_TAG;

  vector<MPI_Comm> c;
  c.resize(maxcolor);

  vector<int> leaders(maxcolor, -1);
  vector<int> newleaders(maxcolor, -1);

  if(rank == 0)
    leaders[color] = global_rank;
  MPI_Allreduce(leaders.data(), newleaders.data(), leaders.size(), MPI_INTEGER, MPI_MAX, global_comm);

  int nSolvers(-1);
  for(int i=0; i<maxcolor; i++) {
    if(i != color && newleaders[i] >= 0) {
      // create a communicator between me and program i
      int tag;
      if(color < i)
        tag = maxcolor*(color+1)+i+1;
      else
        tag = maxcolor*(i+1)+color+1;
      fprintf(stderr,"solver 3: I am here, i = %d, newleaders = %d, tag = %d.\n", i, newleaders[i], tag);
      MPI_Intercomm_create(comm, 0, global_comm, newleaders[i], tag, &(c[i]));
      nSolvers = i+1; //tracks the number of colors/solver
    }
  }

  assert(nSolvers<=maxcolor);
  vector<int> numSolverProcs(maxcolor, 0);
  for(int i=0; i<nSolvers; i++) {
    if(i == color)
      numSolverProcs[i] = size;
    else
      MPI_Comm_remote_size(c[i], &numSolverProcs[i]);
  }

/*
  int joint_rank, joint_size;
  MPI_Comm_rank(joint_comm, &joint_rank);
  MPI_Comm_size(joint_comm, &joint_size);

  fprintf(stderr,"Solver 3 (proc %d of %d) -- joint_comm: %d of %d.\n", rank, size, joint_rank, joint_size);

  int buf = -1;
  if(rank==0) {
    MPI_Recv(&buf, 1, MPI_INT, MPI_ANY_SOURCE, 999, joint_comm, MPI_STATUS_IGNORE);
    fprintf(stderr,"Solver 3 (proc %d of %d): Got code %d!\n", rank, size, buf);
  }
*/

  MPI_Finalize();
  return 0;
}
