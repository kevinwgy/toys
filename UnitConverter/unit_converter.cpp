#include<iostream>
#include<fstream>
#include<string>
#include<stdlib.h>
#include<vector>
using namespace std;

#include <filesystem>
namespace fs = std::filesystem;


template<class T> void WriteDataToFile(std::fstream& file, vector<T> &X);
template<class T> void GetDataInFile(std::fstream& file, vector<T> &X, int MaxCount, bool non_negative);

int main(int argc, char* argv[]) {

  if(argc!=5) {
    fprintf(stdout,"Usage: [UnitConverter]  [path to source files folder]  [source files prefix]  [multiplier]"
                   "  [destination folder]\n");
    exit(-1);
  }


  string path = argv[1];
  if(path.back() == '/') //remove the '/' at the end.
    path = path.substr(0, path.size()-1); 

  string destination_path = argv[4];
  if(destination_path.back() == '/') //remove the '/' at the end.
    destination_path = destination_path.substr(0, destination_path.size()-1); 

  if(!path.compare(destination_path)) {
    fprintf(stderr,"Error: Source and destination folders cannot be the same.\n");
    exit(-1);
  }

  string prefix = argv[2];

  double multiplier = atof(argv[3]);
  
  cout << endl;
  cout << "---------------------------------" << endl;
  cout << "- Source folder: " << path << endl;
  cout << "- Destination folder: " << destination_path << endl;
  cout << "- File(s) prefix/identifier: " << prefix << endl;
  cout << "- Multiplier: " << multiplier << endl;
  cout << "---------------------------------" << endl;

  int MaxCount = 10000000;
  int counter = 0;
  fstream infile, outfile;

  for (const auto & entry : fs::directory_iterator(path)) {

    string full_path_to_file = entry.path();
    string filename = full_path_to_file.substr(path.size()+1, full_path_to_file.size() - path.size() - 1);

    if(prefix.compare(0, prefix.size(), filename, 0, prefix.size()))
      continue;

    vector<double> V;

    infile.open(full_path_to_file.c_str(), std::fstream::in);
    GetDataInFile(infile, V, MaxCount, false);
    infile.close();

    for(int i=0; i<V.size(); i++)
      V[i] *= multiplier;

    string full_path_to_outfile = destination_path + "/" + filename;
    outfile.open(full_path_to_outfile.c_str(), std::fstream::out);
    WriteDataToFile(outfile, V);
    outfile.close();
    
    std::cout << "o Converted data in file: " << filename << " (" << V.size() << " numbers)." << endl;
    counter++;
  }

  cout << "---------------------------------" << endl;
  cout << "- Converted " << counter << " files." << endl;  
  cout << "---------------------------------" << endl;
  cout << "        NORMAL COMPLETION" << endl;
  cout << "---------------------------------" << endl;
  
  return 0;   
}


template<class T>
void GetDataInFile(std::fstream& file, vector<T> &X, int MaxCount, bool non_negative)
{
  double tmp;
  for(int i=0; i<MaxCount; i++) {
    file >> tmp;
    if(file.eof())
      break;
    if(non_negative && tmp<0) {
      fprintf(stderr,"*** Error: Detected negative number (%e) in a data file.\n", tmp);
      exit(-1);
    }
    X.push_back(tmp);
  }
}

template<class T>
void WriteDataToFile(std::fstream& file, vector<T> &X)
{
  for(int i=0; i<X.size(); i++)
    file << std::setprecision(12) << X[i] << endl;
}


