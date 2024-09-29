#include<iostream>
#include<fstream>
#include<cmath>
#include<random>
#include<cassert>
#include<cstdio>
using namespace std;

int main(int argc, char* argv[])
{
  double pi = 2.0*acos(0);
  
  //Define sample size
  int Ntrain(1000), Nvalid(200);
  if(argc==3) {
    Ntrain = atoi(argv[1]);
    Nvalid = atoi(argv[2]);
    assert(Ntrain>0 && Nvalid>0);
  }


  random_device dev0;
  mt19937 rng0(dev0()); //standard generator (engine) seeded with dev()
  uniform_real_distribution<> eps(-0.05, 0.05);

  auto fun = [pi](double x) {return sin(2.0*pi*x);};
  //auto fun = [&](double x) {return -2.0*x + 1.5 + eps(rng0);};

  //random number generators
  double xmax = 1.0;
  random_device dev;
  mt19937 rng(dev()); //standard generator (engine) seeded with dev()
  uniform_real_distribution<> xgen(0.0, xmax);

  FILE* out1 = fopen("training_data.txt","w");
  double x, f;
  for(int i=0; i<Ntrain; i++) {
    x = xgen(rng);
    f = fun(x);
    fprintf(out1,"%14.8e  %14.8e  %4d\n", x, f, f>=0.0?1:0);
  }
  fclose(out1);

  FILE* out2 = fopen("validation_data.txt","w");  
  for(int i=0; i<Nvalid; i++) {
    x = xgen(rng);
    f = fun(x);
    fprintf(out2,"%14.8e  %14.8e  %4d\n", x, f, f>=0.0?1:0);
  }
  fclose(out2);
   
  return 0;
}
