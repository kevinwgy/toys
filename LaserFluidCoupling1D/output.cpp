#include <time.h>
#include "output.h"
#include "input.h"
#include "hgversion.h"
#include <assert.h>
using namespace std;

Output::Output(Input* input) 
{
  sprintf(full_filename_base, "%s/%s", input->file.foldername, input->file.filename_base);
  sprintf(filename_base, "%s", input->file.filename_base);
  char f1[256], f2[256];
  sprintf(f1, "%s_summary.txt", full_filename_base);
  sprintf(f2, "%s_solution.txt", full_filename_base);

  summaryfile.open(f1, ios::out);
  summaryfile << "Computation started at:" << endl;
  summaryfile << "  " << getCurrentDateTime() << endl;
  summaryfile << "Using Code Revision:" << endl;
  summaryfile << "  " << hgRevisionNo << " | " << hgRevisionHash << endl;
  summaryfile << endl;
  summaryfile << "Inputs" << endl;
  summaryfile << "  Xmin                = " << input->file.xmin << endl;
  summaryfile << "  Xmax                = " << input->file.xmax << endl;
  summaryfile << "  NumberOfNodes(N)    = " << input->file.N << endl;
  summaryfile << "  SourceIntensity(I0) = " << input->file.I0 << endl;
  summaryfile << "  Alpha0              = " << input->file.alpha0 << endl;

  double eps = 1.0e-8;
  if(input->file.interval1_xmax - input->file.interval1_xmin > eps) {
    summaryfile << "  Interval1_Xmin      = " << input->file.interval1_xmin << endl;
    summaryfile << "  Interval1_Xmax      = " << input->file.interval1_xmax << endl;
    summaryfile << "  Interval1_alpha     = " << input->file.interval1_alpha << endl;
  }
  if(input->file.interval2_xmax - input->file.interval2_xmin > eps) {
    summaryfile << "  Interval2_Xmin      = " << input->file.interval2_xmin << endl;
    summaryfile << "  Interval2_Xmax      = " << input->file.interval2_xmax << endl;
    summaryfile << "  Interval2_alpha     = " << input->file.interval2_alpha << endl;
  }
  if(input->file.interval3_xmax - input->file.interval3_xmin > eps) {
    summaryfile << "  Interval3_Xmin      = " << input->file.interval3_xmin << endl;
    summaryfile << "  Interval3_Xmax      = " << input->file.interval3_xmax << endl;
    summaryfile << "  Interval3_alpha     = " << input->file.interval3_alpha << endl;
  }

  solfile.open(f2, ios::out);
}

Output::~Output()
{
  if(summaryfile.is_open()) summaryfile.close();
  if(solfile.is_open()) solfile.close();
}


void Output::output_solution(vector<double> &alpha, vector<double> &intensity, double xmin, double xmax, int N)
{
  solfile << "# Node Id  |  X  |  Alpha  |  Intensity" << endl;
  assert(N==alpha.size());
  double dx = (xmax - xmin)/(N-1);
  for(int i=0; i<N; i++) {
    solfile.width(8);
    solfile << i+1 << "  ";
    solfile.precision(8); 
    solfile << scientific << xmin + i*dx << "  ";
    solfile.precision(8); 
    solfile << scientific << alpha[i] << "  ";
    solfile.precision(8); 
    solfile << scientific << intensity[i] << "\n";
  }
  return;
}


// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const string Output::getCurrentDateTime() 
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}
