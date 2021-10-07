#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "input.h"
#include "parser/Assigner.h"
#include "parser/Dictionary.h"

InputFileData::InputFileData() : foldername("./"), filename_base("output")
{
  //TODO: XZ: find physically meaningful values for the defaults of I0, alpha0, xmin, xmax, etc.
  xmin = 0.0;
  xmax = 1.0;
  N = 101;
  I0 = 100.0;
  alpha0 = 10.0;
  interval1_xmin = interval1_xmax = -1e12; interval1_alpha = 10.0;
  interval2_xmin = interval2_xmax = -1e12; interval2_alpha = 10.0;
  interval3_xmin = interval3_xmax = -1e12; interval3_alpha = 10.0;
}


InputFileData::~InputFileData()
{
//  delete[] foldername;
//  delete[] filename_base;
}


void InputFileData::setup(const char *name)
{
  ClassAssigner *ca = new ClassAssigner(name, 16, 0); //father);

  new ClassDouble<InputFileData> (ca, "Xmin", this, &InputFileData::xmin);
  new ClassDouble<InputFileData> (ca, "Xmax", this, &InputFileData::xmax);
  new ClassInt<InputFileData>    (ca, "NumberOfNodes", this, &InputFileData::N);
  new ClassDouble<InputFileData> (ca, "SourceIntensity", this, &InputFileData::I0);

  new ClassDouble<InputFileData> (ca, "Alpha0", this, &InputFileData::alpha0);

  new ClassDouble<InputFileData> (ca, "Interval1_Xmin", this, &InputFileData::interval1_xmin);
  new ClassDouble<InputFileData> (ca, "Interval1_Xmax", this, &InputFileData::interval1_xmax);
  new ClassDouble<InputFileData> (ca, "Interval1_Alpha", this, &InputFileData::interval1_alpha);

  new ClassDouble<InputFileData> (ca, "Interval2_Xmin", this, &InputFileData::interval2_xmin);
  new ClassDouble<InputFileData> (ca, "Interval2_Xmax", this, &InputFileData::interval2_xmax);
  new ClassDouble<InputFileData> (ca, "Interval2_Alpha", this, &InputFileData::interval2_alpha);

  new ClassDouble<InputFileData> (ca, "Interval3_Xmin", this, &InputFileData::interval3_xmin);
  new ClassDouble<InputFileData> (ca, "Interval3_Xmax", this, &InputFileData::interval3_xmax);
  new ClassDouble<InputFileData> (ca, "Interval3_Alpha", this, &InputFileData::interval3_alpha);

  new ClassStr<InputFileData> (ca, "ResultFolder", this, &InputFileData::foldername);
  new ClassStr<InputFileData> (ca, "ResultFilePrefix", this, &InputFileData::filename_base);
}


//-----------------------------------------------------

void Input::readCmdLine(int argc, char** argv)
{
  if(argc==1) {
    fprintf(stderr,"ERROR: Input file not provided!\n");
    exit(-1);
  }
  cmdFileName = argv[1];
}


void Input::readCmdFile()
{
  extern FILE *yyCmdfin;
  extern int yyCmdfparse();

  setupCmdFileVariables();
//  cmdFilePtr = freopen(cmdFileName, "r", stdin);
  yyCmdfin = cmdFilePtr = fopen(cmdFileName, "r");

  if (!cmdFilePtr) {
    fprintf(stderr,"*** Error: could not open \'%s\'\n", cmdFileName);
    exit(-1);
  }

  int error = yyCmdfparse();
  if (error) {
    fprintf(stderr,"*** Error: command file contained parsing errors\n");
    exit(error);
  }
  fclose(cmdFilePtr);
}


void Input::setupCmdFileVariables()
{
  file.setup("LaserFluid1D");
}
