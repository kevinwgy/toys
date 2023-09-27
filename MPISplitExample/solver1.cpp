#include<petscdmda.h>
#include<vector>
using namespace std;

#define M2C_COMM_TAG 0

MPI_Comm comm0;

int main(int argc, char *argv[])
{

  MPI_Init(NULL,NULL);

  MPI_Comm global_comm = MPI_COMM_WORLD;
  int global_rank(0), global_size(0);
  MPI_Comm_rank(global_comm, &global_rank);
  MPI_Comm_size(global_comm, &global_size);

  MPI_Comm comm;
  MPI_Comm_split(global_comm, M2C_COMM_TAG+1, global_rank, &comm);

  comm0 = comm;
  
  PETSC_COMM_WORLD = comm;
  PetscInitialize(&argc, &argv, argc>=3 ? argv[2] : (char*)0, (char*)0);

  int rank(0), size(0);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  fprintf(stderr,"I am proc %d (of %d) --- global %d (of %d): running solver 1.\n", rank, size, global_rank, global_size);


  int m2c_color = M2C_COMM_TAG;
  int maxcolor = 4;

  vector<MPI_Comm> c;
  c.resize(maxcolor);

  vector<int> leaders(maxcolor, -1);
  vector<int> newleaders(maxcolor, -1);

  if(rank == 0) {
    leaders[m2c_color] = global_rank;
  }
  MPI_Allreduce(leaders.data(), newleaders.data(), maxcolor, MPI_INTEGER, MPI_MAX, global_comm);

  for(int i=0; i<maxcolor; i++) {
    if(i != m2c_color && newleaders[i] >= 0) {
      // create a communicator between m2c and program i
      int tag;
      if(m2c_color < i)
        tag = maxcolor * (m2c_color + 1) + i + 1;
      else
        tag = maxcolor * (i + 1) + m2c_color + 1;
      fprintf(stderr,"solver 1: I am here, i = %d, newleaders = %d, tag = %d.\n", i, newleaders[i], tag);
      MPI_Intercomm_create(comm, 0, global_comm, newleaders[i], tag, &(c[i]));
    }
  }

  int AEROS_COLOR = 1;
  MPI_Comm joint_comm = c[AEROS_COLOR];


  int numAerosProcs;
  MPI_Comm_remote_size(joint_comm, &numAerosProcs);

  fprintf(stderr,"Solver 1 (proc %d of %d) -- remote size = %d.\n",
          rank, size, numAerosProcs);

  int joint_rank, joint_size;
  MPI_Comm_rank(joint_comm, &joint_rank);
  MPI_Comm_size(joint_comm, &joint_size);

  fprintf(stderr,"Solver 1 (proc %d of %d) -- joint_comm: %d of %d.\n", rank, size, joint_rank, joint_size);

  if(rank==0)
    MPI_Send(&global_rank, 1, MPI_INT, 0, 999, joint_comm);


  PetscFinalize();
  MPI_Finalize();
  return 0;
}
