#ifndef _INPUT_H_
#define _INPUT_H_
#include <stdio.h>
using namespace std;

struct InputFileData
{
  // computational domain
  double xmin, xmax;
  int N; //number of nodes

  // boundary condition
  double I0; //laser intensity at xmin

  // material parameters
  double alpha0; //baseline
  double interval1_xmin, interval1_xmax, interval1_alpha;
  double interval2_xmin, interval2_xmax, interval2_alpha;
  double interval3_xmin, interval3_xmax, interval3_alpha;

  // output
  const char* foldername;
  const char* filename_base;

  InputFileData();
  ~InputFileData();

  void setup(const char *);
};


class Input
{
  char *cmdFileName;
  FILE *cmdFilePtr;

public:

  InputFileData  file;
  
public:

  Input() {}
  ~Input() {}

  void readCmdLine(int, char**);
  void readCmdFile();
  void setupCmdFileVariables();
  
};
#endif
