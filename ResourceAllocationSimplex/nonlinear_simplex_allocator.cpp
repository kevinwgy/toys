#include<iostream>
#include<vector>
#include<cfloat> //DBL_MAX
#include<cassert>
#include<algorithm>
#include<list>
#include<iomanip>
using namespace std;

double epsilon = 1.0e-15;

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

protected:
  double S(int N) {
    if(law == PERFECT) return (double)N;
    else if(law == AMDAHL) return 1.0/(1.0 - law_coeff + law_coeff/(double)N);
    return (1.0-N)*law_coeff + (double)N;
  }
};


double CostFunction(vector<Solver> &S, vector<int> &n, vector<int> &limiting_solver,
                    vector<int> &non_limiting_solver) {

  assert(S.size() == n.size());
  limiting_solver.clear();

  double max_cost = -DBL_MAX; 
  double my_cost;

  for(int i=0; i<(int)S.size(); i++) {
    my_cost = S[i].Cost(n[i]);
    if(my_cost > max_cost+epsilon) {
      max_cost = my_cost;
      limiting_solver.clear();
      limiting_solver.push_back(i);
    } 
    else if (my_cost >= max_cost-epsilon) {//essentially equal
      limiting_solver.push_back(i);
    }
  }

  vector<bool> tmp(n.size(), 0);
  for(auto&& s : limiting_solver)
    tmp[s] = 1;
  non_limiting_solver.clear();
  for(int i=0; i<(int)tmp.size(); i++)
    if(!tmp[i])
      non_limiting_solver.push_back(i);

  return max_cost;
}


int main(int argc, char* argv[])
{

  //---------------------------------------
  // User inputs
  int N = 50000;
  vector<Solver> solver;
  solver.push_back(Solver(0.2/*dt*/, 0.2/*walltime per step*/, Solver::AMDAHL, 0.999));
  solver.push_back(Solver(1.2/*dt*/, 0.5/*walltime per step*/, Solver::GUSTAFSON, 0.1));
  solver.push_back(Solver(0.2/*dt*/, 1.0/*walltime per step*/, Solver::AMDAHL, 0.9995));
  solver.push_back(Solver(0.1/*dt*/, 0.5/*walltime per step*/, Solver::AMDAHL, 1.0));
  solver.push_back(Solver(0.3/*dt*/, 0.8/*walltime per step*/, Solver::GUSTAFSON, 0.2));
  solver.push_back(Solver(0.4/*dt*/, 1.2/*walltime per step*/, Solver::GUSTAFSON, 0.1));
  solver.push_back(Solver(0.06/*dt*/, 4.5/*walltime per step*/, Solver::GUSTAFSON, 0.05));
  
  //solver.push_back(Solver(0.01/*dt*/, 1.0/*walltime per step*/, Solver::AMDAHL, 1.0));
  //solver.push_back(Solver(0.1/*dt*/, 1.0/*walltime per step*/, Solver::AMDAHL, 1.0));
  //solver.push_back(Solver(0.1/*dt*/, 1.0/*walltime per step*/, Solver::AMDAHL, 1.0));
  //solver.push_back(Solver(0.1/*dt*/, 1.0/*walltime per step*/, Solver::AMDAHL, 1.0));
  //solver.push_back(Solver(0.1/*dt*/, 1.0/*walltime per step*/, Solver::AMDAHL, 1.0));
  //---------------------------------------



  //initialization
  int Nc = solver.size();
  assert(N>=Nc);
  vector<int> allocation(Nc,N/Nc); //solution
  vector<int> limiting_solver, non_limiting_solver;
  allocation[0] += N-(N/Nc)*Nc;
  double cost = CostFunction(solver, allocation, limiting_solver, non_limiting_solver);
  double initial_cost = cost;
  cout << "- Initial assignment:";
  for(int i=0; i<(int)allocation.size(); i++)
    cout << " " << allocation[i] << "(" << solver[i].Cost(allocation[i]) << ")";
  cout << endl;
  cout << "- Initial cost: " << std::setprecision(10) << cost << endl;
  cout << "  o Limited by solver(s):";
  for(auto&& s : limiting_solver)
    cout << " " << s+1;
  cout << endl;


  int maxIt = 100000;
  int iter = 0; 
  vector<int> allocation1(allocation.size(),0);
  list<pair<int, double> > new_cost;
  for(iter = 0; iter < maxIt; iter++) {

    allocation1 = allocation;
     
    double reduction = DBL_MAX;
    for(auto&& i : limiting_solver) {
      reduction = min(reduction, -solver[i].CostVariation(allocation1[i], allocation1[i]+1));
      allocation1[i]++; //the ONLY way to reduce cost function
    }
    double new_max = cost - reduction; //should not exceed new_max

    // take away from the non-limiting solver
    int remaining = limiting_solver.size();
    new_cost.clear();
    for(auto&& i : non_limiting_solver)
      new_cost.push_back(make_pair(i,solver[i].Cost(allocation1[i]-1)));
    new_cost.sort([](pair<int,double> &left, pair<int,double> &right) {return left.second < right.second;}); //sort

    while(remaining>0) {
      if(new_cost.front().second>=new_max)
        break;
      int i = new_cost.front().first;
      allocation1[i]--;
      remaining--;
      new_cost.front().second = solver[i].Cost(allocation1[i]-1);
      //re-sort
      auto it = new_cost.begin();
      for(it = new_cost.begin(); it != new_cost.end(); it++)
        if(new_cost.front().second<=it->second)
          break; 
      if(it != new_cost.begin()) {
        it--;
        if(it != new_cost.begin())
          swap(new_cost.front(), *it);
      }
    }
    if(remaining>0) //no way to decrease cost function
      break;

    allocation = allocation1;
    cost = CostFunction(solver, allocation, limiting_solver, non_limiting_solver);

    cout << "- Iteration " << iter+1 << ": Cost = " << std::setprecision(10) << cost << endl;
    cout << "  o Assignment:";
    for(int i=0; i<(int)allocation.size(); i++)
      cout << " " << allocation[i] << "(" << solver[i].Cost(allocation[i]) << ")";
    cout << endl;
    cout << "  o Limited by solver(s):";
    for(auto&& s : limiting_solver)
      cout << " " << s+1;
    cout << endl;
  }



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
  cout << "  o Limited by solver(s):";
  for(auto&& s : limiting_solver)
    cout << " " << s+1;
  cout << endl;
  cout << "================================================" << endl;

  return 0;
}



