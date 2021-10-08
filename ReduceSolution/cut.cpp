#include<iostream>
#include<fstream>
#include<string>
using namespace std;

int main(int argc, char* argv[])
{
  if(argc!=4 && argc!=5) {
    fprintf(stderr,"Usage: <binary> [input file] [output file] [frequency] | (optional) #Samples\n");
    exit(-1);
  }
  ifstream infile(argv[1],ios::in);
  ofstream outfile(argv[2],ios::out);
  int freq = atoi(argv[3]);
  int N;
  string line;

  getline(infile,line);
  outfile << line << endl;

  if(argc==5) {
    N = atoi(argv[4]);
    outfile << N << endl;
  }
  else {
    infile >> N;
    outfile << N << endl;
    getline(infile,line); //nothing here...
  }

  fprintf(stderr,"N = %d.\n", N);
  int count = 0;
  getline(infile,line);
  while(1) {
    bool write = !(count%freq);
    if(write) outfile << line << endl;
    for(int i=0; i<N; i++) {
      if(i==0 && write) cout << "Writing solution at t = " << line << endl;
      getline(infile,line);
      if(write) outfile << line << endl;
    }
    count++;
    getline(infile,line);
    if(infile.eof()) break;
  }
  fprintf(stderr,"DONE!\n");
  return 0;
}
