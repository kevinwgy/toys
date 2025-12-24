/************************************************************************
 * Copyright © 2020 The Multiphysics Modeling and Computation (M2C) Lab
 * <kevin.wgy@gmail.com> <kevinw3@vt.edu>
 ************************************************************************/

#ifndef _IO_DATA_H_
#define _IO_DATA_H_

#include <cstdio>
#include <map>
#include <parser/Assigner.h>
#include <parser/Dictionary.h>

using std::map;

/*********************************************************************
 * class IoData reads and processes the input data provided by the user
 *********************************************************************
*/
//------------------------------------------------------------------------------

template<class DataType>
class ObjectMap {

public:
  map<int, DataType *> dataMap;

  void setup(const char *name, ClassAssigner *p) {
    SysMapObj<DataType> *smo = new SysMapObj<DataType>(name, &dataMap);
    if (p) p->addSmb(name, smo);
    else addSysSymbol(name, smo);
  }

  ~ObjectMap() {
    for(typename map<int, DataType *>::iterator it=dataMap.begin();it!=dataMap.end();++it)
      delete it->second;
  }
};

//------------------------------------------------------------------------------

struct SolverData {

  enum Law {PERFECT = 0, AMDAHL = 1, GUSTAFSON = 2} law;
  double law_coeff;
  double dt; //physical dt
  double tprime; //wall time on one proc;

  SolverData();
  ~SolverData() {}

  Assigner *getAssigner();
};

//------------------------------------------------------------------------------

struct InputData {

  ObjectMap<SolverData> solverMap;
  int N;

  InputData();
  ~InputData() {}

  void setup(const char *, ClassAssigner * = 0);
};

//------------------------------------------------------------------------------

class IoData {

  char *cmdFileName;
  FILE *cmdFilePtr;

public:

  InputData input;

public:

  IoData() {}
  IoData(int, char**);
  ~IoData() {}

  void readCmdLine(int, char**);
  void setupCmdFileVariables();
  void readCmdFile();

};

//------------------------------------------------------------------------------
#endif
