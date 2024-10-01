#include<iostream>
#include<fstream>
#include<cmath>
#include<random>
#include<cassert>
#include<cstdio>
using namespace std;

int main(int argc, char* argv[])
{
  //Define sample size
  int Ntrain(1000), Nvalid(200);
  if(argc==3) {
    Ntrain = atoi(argv[1]);
    Nvalid = atoi(argv[2]);
    assert(Ntrain>0 && Nvalid>0);
  }

  auto fun = [](double x, double y) {return x*y;};

  //random number generators
  double xmax = 10.0, ymax = 10.0;
  random_device dev;
  mt19937 rng(1); //standard generator (engine) seeded with a fixed number (1)
  //mt19937 rng(dev()); //standard generator (engine) seeded with dev()
  uniform_real_distribution<> xgen(-xmax, xmax);
  uniform_real_distribution<> ygen(-ymax, ymax);

  FILE* out1 = fopen("training_data.txt","w");
  double x,y,f;
  for(int i=0; i<Ntrain; i++) {
    x = xgen(rng);
    y = ygen(rng);
    f = fun(x,y);
    fprintf(out1,"%14.8e  %14.8e  %14.8e  %4d\n", x, y, f, f>=0.0?1:0);
  }
  fclose(out1);

  FILE* out2 = fopen("validation_data.txt","w");  
  for(int i=0; i<Nvalid; i++) {
    x = xgen(rng);
    y = ygen(rng);
    f = fun(x,y);
    fprintf(out2,"%14.8e  %14.8e  %14.8e  %4d\n", x, y, f, f>=0.0?1:0);
  }
  fclose(out2);
   
  return 0;
}
