/************************************************************************
 * Copyright © 2020 The Multiphysics Modeling and Computation (M2C) Lab
 * <kevin.wgy@gmail.com> <kevinw3@vt.edu>
 ************************************************************************/

#include <IoData.h>
#include <parser/Assigner.h>
#include <parser/Dictionary.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cfloat>
#include <climits>
#include <cmath>
#include <unistd.h>
#include <bits/stdc++.h> //INT_MAX
//#include <dlfcn.h>
using namespace std;

RootClassAssigner *nullAssigner = new RootClassAssigner;

//------------------------------------------------------------------------------

SolverData::SolverData()
{
  law = PERFECT;
  law_coeff = 0.0;
  dt = 1.0;
  tprime = 1.0;
}

//------------------------------------------------------------------------------

//void SolverData::setup(const char *name, ClassAssigner *father)
Assigner *SolverData::getAssigner()
{
  ClassAssigner *ca = new ClassAssigner("normal", 4, nullAssigner);

  new ClassToken<SolverData> (ca, "ScalingLaw", this,
      reinterpret_cast<int SolverData::*>(&SolverData::law), 3,
      "Perfect", 0, "Amdahl", 1, "Gustafson", 2);
  new ClassDouble<SolverData>(ca, "ScalingLawCoefficient", this, &SolverData::law_coeff);
  new ClassDouble<SolverData>(ca, "PhysicalTimeStep", this, &SolverData::dt);
  new ClassDouble<SolverData>(ca, "WallTimePerStepOneCore", this, &SolverData::tprime);

  return ca;
}

//------------------------------------------------------------------------------

InputData::InputData()
{
  N = 100;
}

//------------------------------------------------------------------------------

void InputData::setup(const char *name, ClassAssigner *father)
{
  ClassAssigner *ca = new ClassAssigner(name, 1, father);
  new ClassInt<InputData>(ca, "Processors", this, &InputData::N);
  solverMap.setup("Solver", ca);
}

//------------------------------------------------------------------------------

IoData::IoData(int argc, char** argv)
{
  readCmdLine(argc, argv);
  readCmdFile();
}

//------------------------------------------------------------------------------

void IoData::readCmdLine(int argc, char** argv)
{
  if(argc==1) {
    fprintf(stdout,"\033[0;31m*** Error: Input file not provided!\n\033[0m");
    exit(-1);
  }
  cmdFileName = argv[1];
}

//------------------------------------------------------------------------------

void IoData::readCmdFile()
{
  extern FILE *yyCmdfin;
  extern int yyCmdfparse();

  setupCmdFileVariables();
//  cmdFilePtr = freopen(cmdFileName, "r", stdin);
  yyCmdfin = cmdFilePtr = fopen(cmdFileName, "r");

  if (!cmdFilePtr) {
    fprintf(stdout,"\033[0;31m*** Error: could not open \'%s\'\n\033[0m", cmdFileName);
    exit(-1);
  }

  int error = yyCmdfparse();
  if (error) {
    fprintf(stdout,"\033[0;31m*** Error: command file contained parsing errors.\n\033[0m");
    exit(error);
  }
  fclose(cmdFilePtr);
}

//------------------------------------------------------------------------------

void IoData::setupCmdFileVariables()
{
  input.setup("Input");  
}

//------------------------------------------------------------------------------
