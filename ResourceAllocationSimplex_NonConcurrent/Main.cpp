#include<iostream>
#include<vector>
#include<cfloat> //DBL_MAX
#include<cmath>
#include<numeric>
#include<cassert>
#include<algorithm>
#include<list>
#include<iomanip>
#include<IoData.h>
using namespace std;

double epsilon = 1.0e-15;

//-----------------------------------------------------------------------

class Solver {

public:
  enum Law {PERFECT = 0, AMDAHL = 1, GUSTAFSON = 2};

protected:
  Law law;
  double law_coeff;
  double dt; //physical dt
  double tprime; //wall time on one proc;

public:
  Solver(double dt_, double tprime_, Law law_, double law_coeff_ = 0.0) : law(law_), law_coeff(law_coeff_),
         dt(dt_), tprime(tprime_) {}

  double Cost(int N) {return tprime/(dt*S(N));}
  double CostVariation(int N, int M) {return Cost(M) - Cost(N);}
  double GetTPrime() {return tprime;}

protected:
  double S(int N) {
    if(law == PERFECT) return (double)N;
    else if(law == AMDAHL) return 1.0/(1.0 - law_coeff + law_coeff/(double)N);
    return (1.0-N)*law_coeff + (double)N;
  }
};

//-----------------------------------------------------------------------
double CostFunction(vector<Solver> &S, vector<int> &n);

//-----------------------------------------------------------------------

int main(int argc, char* argv[])
{

  IoData iod(argc, argv);

  //---------------------------------------
  // User inputs
  int N = iod.input.N;
  vector<Solver> solver;
  int Nc = iod.input.solverMap.dataMap.size(); //total number of solvers
  assert(Nc<=N);
  for(auto&& s : iod.input.solverMap.dataMap) {
    int id = s.first;
    if(id<0 || id>=Nc) {
      fprintf(stdout, "*** Error: Solver id should be between 0 and %d. Found %d.\n", Nc-1, id);
      exit(-1);
    }

    Solver::Law mylaw;
    if(s.second->law == SolverData::PERFECT)        mylaw = Solver::PERFECT;
    else if(s.second->law == SolverData::AMDAHL)    mylaw = Solver::AMDAHL;
    else if(s.second->law == SolverData::GUSTAFSON) mylaw = Solver::GUSTAFSON;
    else {
      fprintf(stdout, "*** Error: Found unknown scaling law.\n");
      exit(-1);
    }
    solver.push_back(Solver(s.second->dt, s.second->tprime, mylaw, s.second->law_coeff));
  }

  //initialization
  vector<int> allocation(Nc,N/Nc); //solution
  allocation[0] += N-(N/Nc)*Nc;
  double cost = CostFunction(solver, allocation);
  double initial_cost = cost;
  cout << "- Initial assignment:";
  for(int i=0; i<(int)allocation.size(); i++)
    cout << " " << allocation[i] << "(" << solver[i].Cost(allocation[i]) << ")";
  cout << endl;
  cout << "- Initial cost: " << std::setprecision(10) << cost << endl;
  cout << endl;


  int maxIt = 100000;
  int iter = 0; 
  vector<int> allocation1(allocation.size(),0);
  for(iter = 0; iter < maxIt; iter++) {

    allocation1 = allocation;
     
    double var = -1.0;
    int var_index = -1;
    for(int i=0; i<Nc; i++) {
      double my_var = -solver[i].CostVariation(allocation1[i], allocation1[i]+1);
      if(var < my_var) {
        var = my_var;
        var_index = i;
      } 
    }

    double new_cost = cost - var;
    allocation1[var_index]++;

    var = -1.0;
    int var_index_2 = -1;
    for(int i=0; i<Nc; i++) {
      if(i==var_index)
        continue;
      double my_var = solver[i].CostVariation(allocation1[i], allocation1[i]-1);
      if(var < my_var) {
        var = my_var;
        var_index_2 = i;
      } 
    }

    new_cost += var;
    allocation1[var_index_2]--;

    if(new_cost >= cost)
      break; //no way to reduce cost

    allocation = allocation1;
    cost = new_cost;

    cout << "- Iteration " << iter+1 << ": Cost = " << std::setprecision(10) << cost << endl;
    cout << "  o Assignment:";
    for(int i=0; i<(int)allocation.size(); i++)
      cout << " " << allocation[i] << "(" << solver[i].Cost(allocation[i]) << ")";
    cout << endl;
  }

  // Now, compute the cost of a naive assignment for comparison only. In this assignment, we
  // split resources based on wall time per time step on one core (i.e., tprime)
  vector<int> naive_allocation(Nc,0);
  double tprime_sum = 0.0;
  for(auto&& s : solver)
    tprime_sum += s.GetTPrime();
  for(int i=0; i<Nc; i++)
    naive_allocation[i] += (int)round(solver[i].GetTPrime()/tprime_sum*N);
  int remainder = N - accumulate(naive_allocation.begin(), naive_allocation.end(), 0);
  if(remainder != 0) {
    for(int i=0; i<Nc; i++) {
      if(remainder>0) {
        naive_allocation[i]++;
        remainder--;
      } else if(remainder<0) {
        naive_allocation[i]--;
        remainder++;
      } else
        break;
    }
  }
  assert(remainder==0);
  double naive_cost = CostFunction(solver, naive_allocation);
  cout << endl;
  cout << "- Naive assignment:";
  for(int i=0; i<(int)naive_allocation.size(); i++)
    cout << " " << naive_allocation[i] << "(" << solver[i].Cost(naive_allocation[i]) << ")";
  cout << endl;
  cout << "- Naive assignment cost: " << std::setprecision(10) << naive_cost << endl;
  cout << endl;
  cout << endl;
 

  cout << "================================================" << endl;
  if(iter<maxIt)
    cout << "Converged in " << iter+1 << " iterations." << endl;
  else 
    cout << "Did not converge." << endl;
  cout << "Optimal assignment:";
  for(int i=0; i<(int)allocation.size(); i++)
    cout << " " << allocation[i] << "(" << solver[i].Cost(allocation[i]) << ")";
  cout << endl;
  cout << "Final cost: " << std::setprecision(10) << cost << endl;
  cout << "Speedup v.s. initial assignment: " << initial_cost/cost << "X" << endl;
  cout << "Speedup v.s. `naive' assignment: " << naive_cost/cost << "X" << endl;
  cout << "================================================" << endl;

  return 0;
}

//-----------------------------------------------------------------------

double CostFunction(vector<Solver> &S, vector<int> &n) {

  assert(S.size() == n.size());

  double cost = 0.0;
  for(int i=0; i<(int)S.size(); i++)
    cost += S[i].Cost(n[i]);

  return cost;
}

//-----------------------------------------------------------------------

