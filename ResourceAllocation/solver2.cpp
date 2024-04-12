#include<mpi.h>
#include<unistd.h> //sleep
#include<iomanip>
#include<vector>
#include<cassert>
#include<string>
#include<cmath> //floor
#include<chrono>
using namespace std;

#define SOLVER_COMM_TAG 1
#define MAX_NUM_COLOR 10

int main(int argc, char *argv[])
{
  if(argc!=4) {
    fprintf(stdout,"Usage: [binary] [physical time step size] [computation time (in seconds) for each step] "
                   "[final physical time].\n");
    exit(-1);
  }

  int maxcolor = MAX_NUM_COLOR;
  int color = SOLVER_COMM_TAG; //color --> solver

  double dt = atof(argv[1]); //physical time step
  double walldt = atof(argv[2]); //computation time for each time step
  double tfinal = atof(argv[3]); //final physical time

  MPI_Init(NULL,NULL);

  int global_rank(0), global_size(0);
  MPI_Comm global_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(global_comm, &global_rank);
  MPI_Comm_size(global_comm, &global_size);

  //Make sure tfinal is the same in all solvers
  double tfinal_max = tfinal;
  MPI_Allreduce(MPI_IN_PLACE, &tfinal_max, 1, MPI_DOUBLE, MPI_MAX, global_comm);
  if(tfinal <= 0 || fabs(tfinal-tfinal_max)/fabs(tfinal)>1.0e-10) {
    fprintf(stderr,"[Proc %d]: Incorrect or inconsistent final time (%e/%e).\n", tfinal, tfinal_max);
    fflush(stderr);
    MPI_Finalize();
    exit(-1);
  }

  // Split MPI_COMM_WORLD to get intracomms
  MPI_Comm comm;
  MPI_Comm_split(global_comm, color+1, global_rank, &comm);

  int rank(0), size(0);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  fprintf(stdout,"[Proc %d]: Running Solver %d (Rank %d/%d).\n",
          global_rank, color+1, rank, size);
  fflush(stdout);
  MPI_Barrier(global_comm);


  // Build solver-solver intercomms
  vector<MPI_Comm> c;
  c.resize(maxcolor);
  vector<int> leaders(maxcolor, -1);
  vector<int> newleaders(maxcolor, -1);

  if(rank == 0)
    leaders[color] = global_rank;
  MPI_Allreduce(leaders.data(), newleaders.data(), leaders.size(), MPI_INTEGER, MPI_MAX, global_comm);

  for(int i=0; i<maxcolor; i++) {
    if(i != color && newleaders[i] >= 0) {
      // create a communicator between me and program i
      int tag;
      if(color < i)
        tag = maxcolor*(color+1)+i+1;
      else
        tag = maxcolor*(i+1)+color+1;
      MPI_Intercomm_create(comm, 0, global_comm, newleaders[i], tag, &(c[i]));
    }
  }

  
  // Get dt from all Solvers
  vector<double> dt_all(maxcolor,0.0);
  dt_all[color] = dt;
  MPI_Allreduce(MPI_IN_PLACE, dt_all.data(), dt_all.size(), MPI_DOUBLE, MPI_MAX, global_comm);

  vector<int> numSolverProcs(maxcolor, 0);
  for(int i=0; i<maxcolor; i++) {
    if(i == color)
      numSolverProcs[i] = size;
    else if(newleaders[i]>=0) {
      MPI_Comm_remote_size(c[i], &numSolverProcs[i]);
      if(rank==0)
        fprintf(stdout,"[Proc %d]: Leader of Solver %d (size: %d, dt: %e). "
                "Built intercomm w/ Solver %d (size: %d, dt: %e).\n",
                global_rank, color+1, size, dt_all[color], i+1, numSolverProcs[i], dt_all[i]);
    }
  }

  fflush(stdout);
  MPI_Barrier(global_comm);


  // Artificial time loop
  if(rank == 0)
    fprintf(stdout,"[Proc %d]: Solver %d enters main loop.\n", global_rank, color+1);
  fflush(stdout);
  MPI_Barrier(global_comm);

  auto start_time = chrono::system_clock::now();
  double t = 0.0;
  string filename = "output_solver" + to_string(color+1) + ".txt";
  //FILE *out = stdout;
  FILE *out = fopen(filename.c_str(), "w"); //only leader writes to the file
  if(rank == 0) { 
    fprintf(out, "==========================\n");
    fprintf(out, "         SOLVER %d\n", color+1);
    fprintf(out, "==========================\n");
    fprintf(out, "dt = %e, walldt = %e, tfinal = %e.\n", dt, walldt, tfinal);
    fflush(out);
  }

  int step = 0;
  while(t<tfinal-1e-10) {

    //-----------------------------
    // Pretend to be working :P
    double wait_time = walldt/size; //assuming perfect scalability
    usleep(wait_time*1e6);
    //-----------------------------

    //-----------------------------
    // Send/receive data (Only leaders talk)
    //-----------------------------
    if(rank == 0) { //leader of the solver

      auto current_time = chrono::system_clock::now();
      chrono::duration<double> elapsed_seconds = current_time - start_time;
      fprintf(out, "Step %d: t = %e, computation time = %e s.\n", step, t, elapsed_seconds.count());
      fflush(out);

      for(int i=0; i<maxcolor; i++) {
        if(color==i || newleaders[i]<0)
          continue;

        // should I talk to solver i now?
        bool talk = false;
        if(step==0)
          talk = true; //first time-step
        else if(dt_all[i]<=dt) //their dt is smaller or equal, always send
          talk = true;
        else if (floor((t+dt)/dt_all[i]) != floor(t/dt_all[i]))
          talk = true;
        //else
        //  fprintf(out, "  o floor0 = %d, floor1 = %d, t = %e, dt = %e, dt_o = %e.\n",
        //          (int)floor((t-dt)/dt_all[i]), (int)floor(t/dt_all[i]), t, dt, dt_all[i]);
        //if(talk)
        //  fprintf(out, "  o Want to exchange data with Solver %d.\n", i+1);
      }
      fflush(out);

      vector<MPI_Request> send_requests, recv_requests;
      for(int i=0; i<maxcolor; i++) {
        if(color==i || newleaders[i]<0)
          continue;

        // should I talk to solver i now?
        bool talk = false;
        if(step==0)
          talk = true; //first time-step
        else if(dt_all[i]<=dt) //their dt is smaller or equal, always send
          talk = true;
        else if (floor((t+dt)/dt_all[i]) != floor(t/dt_all[i]))
          talk = true;
          
 
        // exchange data
        if(talk) {
          vector<double> sendbuffer(100,0.0); //buffer size = 100
          send_requests.push_back(MPI_Request());
          MPI_Isend(sendbuffer.data(), sendbuffer.size(), MPI_DOUBLE, 0, color, c[i], &send_requests.back());

          vector<double> recvbuffer(100,0.0); //buffer size = 100
          recv_requests.push_back(MPI_Request());
          MPI_Irecv(recvbuffer.data(), recvbuffer.size(), MPI_DOUBLE, MPI_ANY_SOURCE, i, c[i], &recv_requests.back());

          fprintf(out, "  o Exchanging data with Solver %d.\n", i+1);
          fflush(out);
        }
      }
      
      MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE); //not necessary(?)
      MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);
      send_requests.clear();
      recv_requests.clear();
    }

    // advance physical time
    t += dt;
    step++;
  }

  MPI_Barrier(global_comm);

  if(rank == 0) { 
    fprintf(out, "==========================\n");
    fprintf(out, "    Normal Termination\n");
    fprintf(out, "==========================\n");
    auto current_time = chrono::system_clock::now();
    chrono::duration<double> elapsed_seconds = current_time - start_time;
    fprintf(out, "Computation time: %e.\n", elapsed_seconds.count());

    fprintf(stdout,"[Proc %d]: Solver %d --> Normal Termination\n", global_rank, color+1);
    fflush(stdout);
  }
  fclose(out);
  

  MPI_Finalize();
  return 0;
}
