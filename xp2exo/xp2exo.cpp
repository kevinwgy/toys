//-----------------------------------------------------------
//  Description: This routine extracts a mesh and results in
//               XPost format and converts into Exodus format
//               which can then be loaded in Paraview.
//       Inputs: <path to mesh file> <path/s to result file>
//      Outputs: <path to file containing mesh and results>
//       Author: Alex Main
//        Notes: (WARNING) this code doesn't verify inputs.
//-----------------------------------------------------------

/* Exodus Naming Conventions
http://www.paraview.org/Wiki/Restarted_Simulation_Readers#Exodus_Naming_Conventions
By default, ParaView will assume that the numbers at the end of a file represent partition numbers and will attempt to load them all at once. But there is a convention for having numbers specifying a temporal sequence instead. If the files end in .e-s.{RS} (where {RS} is some number sequence), then the numbers (called restart numbers) are taken as a partition in time instead of space. For example, the following sequence of files represents a time series.

mysimoutput.e-s.000
mysimoutput.e-s.001
mysimoutput.e-s.002
*/

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include <sstream>
#include <cstring>
#include <set>
#include <list>
#include <limits>
#include "netcdf.h"
#include "exodusII.h"

using namespace std;

struct elem_block {
  
  int elid;
  int num_elem;
  list<int> elems;
  vector<int>* connect;
  string name;
};

typedef std::map<int, elem_block> top_block;

void writeMesh(int exoid, int nNodes, int num_elem, int num_elem_block, std::map<int, int> &elem_nodes,
               vector<double> &X, vector<double> &Y, vector<double> &Z, vector< top_block* > &myElements,
               set<int> &deletedElements);
void writeGlobalVar(int exoid, int numV, char **names, int *decomposition);
void deleteElements(int &num_elem, vector< top_block* > &myElements, set<int> &deletedElements,
                    std::map<int, int> &elem_nodes);

static void usage() {
  
  std::cout << "usage: xp2exo <top file> <output file (exodus ii)> [<xpost result list>]" << std::endl;
}

int main(int argc, char** argv)
{
  char word1[20],word2[20],word3[20];
  int int1,ncomponent,frame;

  vector< top_block* > myElements;
  set<int> deletedElements;

  double x,y,z;
  double garbage;
  char line[512];
  int CPU_word_size=8,IO_word_size=8;
  int bFictiousTime = 0;

  std::map<int,int> exodusRemap;

  int no_remap = 0;
 
  if (argc < 3) {
    usage();
    return 0;
  }


  if (strcmp(argv[1], "--noremap") == 0) {

    no_remap = 1;
    argc--;
    argv = argv+1;
  }

  //parsing
  ifstream inMesh(argv[1],ios::in);
 
  //preparation
  inMesh >> word1 >> word2;

  int exoid = ex_create (argv[2],/* filename path */
			 EX_CLOBBER,/* create mode */
			 &CPU_word_size,/* CPU float word size in bytes */
			 &IO_word_size);/* I/O float word size in bytes */

  int num_dim = 3;
  vector<double> X,Y,Z;

  //load node coordinates
  while(!inMesh.eof()) {
    inMesh >> int1;
    if(inMesh.fail()) {
      inMesh.clear();
      break;
    }
    
    inMesh >> x >> y >> z;
    X.push_back(x);
    Y.push_back(y);
    Z.push_back(z);
    exodusRemap[ int1 ] = X.size();
  }

  int nNodes = (int)X.size();
  cout<<"Loaded "<< nNodes <<" nodes."<<endl;
  
  std::string name,tmp,tmp1;
  int id,elid,i,en;

  int num_elem_block = 0;
  int num_elem = 0;
  std::map<int, int> elem_nodes;
  elem_nodes[1] = 2;
  elem_nodes[2] = 4;
  elem_nodes[4] = 3;
  elem_nodes[5] = 4;
  elem_nodes[6] = 2;
  elem_nodes[66] = 2;
  elem_nodes[101] = 2;
  elem_nodes[103] = 4;
  elem_nodes[106] = 2;
  elem_nodes[107] = 2;
  elem_nodes[117] = 8;
  elem_nodes[121] = 2;
  elem_nodes[122] = 2;
  elem_nodes[104] = 3;
  elem_nodes[108] = 3;
  elem_nodes[123] = 4;
  elem_nodes[124] = 6;
  elem_nodes[120] = 3;
  elem_nodes[125] = 10;
  elem_nodes[172] = 20;
  elem_nodes[188] = 4;
  elem_nodes[145] = 8;
  elem_nodes[146] = 3;
  elem_nodes[506] = 2;
  elem_nodes[2120] = 4;
  elem_nodes[4746] = 4;

  while(!inMesh.eof()) {
    inMesh.getline(line,512);
    std::stringstream mystrm(line);
    mystrm >> tmp1 >> name >> tmp >> tmp;
    if (tmp1 != "Elements")
      break;
    //inMesh.getline(line,512);

    top_block* top_blocks = new top_block;
    elem_block E;
    E.elid = -1;
    E.name = name;
    E.num_elem = 0;
    E.connect = new vector<int>;
    elem_block* curr = &E;
    while(!inMesh.eof()) {
      inMesh >> id;
      //std::cout << id << std::endl;
      if (inMesh.fail()) {
	if (!inMesh.eof())
	  inMesh.clear();
	break;
      }

      inMesh >> en;
      if (elem_nodes.find(en) == elem_nodes.end()) {
	std::cout << "unknown topology: " << en << std::endl;
	exit(-1);
      }
      /*if (curr->elid == -1) {
	curr->elid = en;
	(*top_blocks)[en] = E;
      }
      else if (en != curr->elid) {*/
	//std::cout << "Error: found elem block with two types of elements!" << std::endl;
	//exit(1);
	top_block::iterator itr = top_blocks->find(en);
	if (itr != top_blocks->end()) {
	  curr = &itr->second;
	} else {
	  E.elid = en;
	  E.name = name;
	  E.num_elem = 0;
	  E.connect = new vector<int>;
	  curr = &((*top_blocks)[en] = E);
	}
	//}

      curr->elems.push_back(id);
      int nn = elem_nodes[curr->elid];
      for (i = 0; i < nn; ++i) {
	inMesh >> en;
	curr->connect->push_back( exodusRemap[en] );
      }
      if(curr->elid == 172) {
        // if element type is 172 need to switch nodes 12-16 with nodes 17-20
        std::swap_ranges(curr->connect->end()-4, curr->connect->end(), curr->connect->end()-8);
      }
      ++num_elem;
    }
    myElements.push_back(top_blocks);
    num_elem_block += top_blocks->size();
  }     

  inMesh.close();

  writeMesh(exoid, nNodes, num_elem, num_elem_block, elem_nodes, X, Y, Z, myElements, deletedElements);

  int m = 3;
  ifstream *sols = new ifstream[argc-3];
  int *ncomp = new int[argc-3];
  char **names = new char*[(argc-3)*6];
  int numV = 0;
  bool *isNodal = new bool[argc-3];
  const char* xyz[6] = {"X","Y", "Z","W","A","B"};
  int* decomposition = 0;
  int* decsize;
  int num_elem_local;
  int deletedElementsFile = -1, deletedElementNo;
  double deletedElementTime = std::numeric_limits<double>::infinity();
  string cause;

  top_block::iterator itr = myElements[0]->begin();
  if(itr != myElements[0]->end()) {
    int nn00 = elem_nodes[itr->second.elid];
    num_elem_local = itr->second.connect->size() / nn00;
  }
  else {
    num_elem_local = 0;
  }
  for (i = 0; i < (argc-3)*6; ++i)
    names[i] = new char[64];
  for (m = 0; m < argc-3; ++m) {
    sols[m].open(argv[m+3],ios::in);
    if (!sols[m].good()) {
      std::cout << "Could not read file " << argv[m+3] << std::endl;
    } else {
      std::cout << "Reading file " << argv[m+3] << std::endl;

    }
    sols[m].getline(line,512);
    if(strcmp(line,"#  time   Element_no   Cause") == 0) {
      if (strcmp(argv[2]+(strlen(argv[2])-2),".e") != 0) {
        std::cerr << "error: output file must have \".e\" extension\n";
        exit(-1);
      }
      string cmd = "rm -f "; cmd += argv[2]; cmd += "-s.*";
      int x = system(cmd.c_str());
      isNodal[m] = false;
      deletedElementsFile = m;
      sols[m] >> deletedElementTime >> deletedElementNo >> cause;
      continue;
    }
    char type[64],name[64];
    int read = sscanf(line,"%s %s",type,name);
    if (read > 0) { // this is a solution file
      ncomponent = (strcmp(type,"Scalar") == 0 ? 1 : 3);
      if (strcmp(type,"Vector5") == 0)
        ncomponent = 5;
      if (strcmp(type,"Vector6") == 0)
        ncomponent = 6;
      std::cout << "Reading variable " << name << " with " << ncomponent << " components" << std::endl;
      if (ncomponent == 3) {
        for (i = 0; i <3; ++i) {
          strcpy(names[numV+i],name);
          strcat(names[numV+i],xyz[i]);
        }
        numV += 3;
      } else if (ncomponent == 5) {
        for (i = 0; i <5; ++i) {
          strcpy(names[numV+i],name);
          strcat(names[numV+i],xyz[i]);
        }
        numV += 5;
      } else if (ncomponent == 6) {
        for (i = 0; i <6; ++i) {
          strcpy(names[numV+i],name);
          strcat(names[numV+i],xyz[i]);
        }
        numV += 6;
      } else {
        strcpy(names[numV],name);
        numV++;
      }
      ncomp[m] = ncomponent;
      isNodal[m] = true;
    } else {
      isNodal[m] = false;
      sols[m].getline(line,512);
      std::cout << "Line = " << line << std::endl;
      int nparts;
      sols[m] >> nparts;
      std::cout << "# of elems (3D) = " << num_elem_local << std::endl;
      decomposition = new int[num_elem_local];
      decsize = new int[nparts];
      int tmp;
      std::cout << "Reading decomposition into " << nparts << " parts" << std::endl;
      for (int j = 0; j < nparts; ++j) {
        sols[m] >> decsize[j];
        for (int l = 0; l < decsize[j]; ++l) {
          sols[m] >> tmp;
          decomposition[tmp-1] = j+1;
        }
      } 
    }
  }   

  if (argc > 3) { 
      //load solution file
      writeGlobalVar(exoid, numV, names, decomposition);
      double *Sol = new double[nNodes*numV];
      double timestamp,dummy;
      int k = 0,l;

      int m0 = 0;
      if (!isNodal[0])
        ++m0;

      int mynn,lid;
      for (m = 0; m < argc-3; ++m) {
        if (isNodal[m])
          sols[m] >> mynn;
      }
      int s=0;
      while (m0 < argc-3 && !sols[m0].eof() && argc > 3) {
        for (m = 0; m < argc-3; ++m) {
          if (isNodal[m])
            sols[m] >> timestamp;
        } 
        if (sols[m0].eof())
	  break;

        if(timestamp > 0 && deletedElementsFile > -1) {
          // check for deleted elements up to the current timestamp
          deletedElements.clear();
          while (deletedElementTime <= timestamp) {
            deletedElements.insert(deletedElementNo);
            sols[deletedElementsFile] >> deletedElementTime >> deletedElementNo >> cause;
            if(sols[deletedElementsFile].eof()) deletedElementTime = std::numeric_limits<double>::infinity();
          }

          // if there are any elements to delete then open a new output file and increment s
          if(deletedElements.size() > 0) {
            deleteElements(num_elem, myElements, deletedElements, elem_nodes);

            char *filename = new char[strlen(argv[2])+10];
            char extension[8];
            sprintf(extension,"-s.%04d",s+1);
            strcpy(filename,argv[2]);
            strcat(filename,extension);
            exoid = ex_create (filename,/* filename path */
                               EX_CLOBBER,/* create mode */
                               &CPU_word_size,/* CPU float word size in bytes */
                               &IO_word_size);/* I/O float word size in bytes */
            delete [] filename;

            writeMesh(exoid, nNodes, num_elem, num_elem_block, elem_nodes, X, Y, Z, myElements, deletedElements);
            writeGlobalVar(exoid, numV, names, decomposition);
            k=0;
            s++;
          }
          std::cout << "Read part " << s << " frame " << k << " Timestamp = " << timestamp << std::endl;
        }
        else {
          std::cout << "Read frame " << k << " Timestamp = " << timestamp << std::endl;
        }
      
        ex_put_time(exoid,k+1,&timestamp);
        if (decomposition) {
          double* decToWrite = new double[num_elem_local];
          for (l = 0; l < num_elem_local; ++l) {
            decToWrite[l] = (double)decomposition[l];
          }
          ex_put_elem_var(exoid, k+1, 1, 1, num_elem_local, decToWrite);
          delete [] decToWrite;
        }  
      
        for(int i=0; i<mynn; i++) {
          l = 0;
	  if (!no_remap && exodusRemap.find(i+1) == exodusRemap.end()) {
            for (m = 0; m < argc-3; ++m) {
              if (isNodal[m]) { 
	        for(int j=0; j<ncomp[m]; j++) 
	          sols[m] >> dummy;
              }
	    
	    }
          } else {
            if (!no_remap)
  	      lid = exodusRemap[i+1];
            else
              lid = i+1;
            for (m = 0; m < argc-3; ++m) {
              if (isNodal[m]) {
	        for(int j=0; j<ncomp[m]; j++,l++) 
	          sols[m] >> Sol[lid-1+nNodes*l];
	      }
            }
	  }
        }

        l = 0;
        for (m = 0; m < argc-3; ++m) {
          if (isNodal[m]) {
            for(int j=0; j<ncomp[m]; j++,l++) 
	      ex_put_nodal_var(exoid, k+1, l+1, nNodes, Sol+l*nNodes);
          }
        }

        
        ex_update (exoid);
        ++k;
      }
      delete [] Sol;
  } 
  // clean-up
  ex_close(exoid);

  delete [] sols;
  delete [] ncomp;
  for (i = 0; i < (argc-3)*6; ++i)
    delete [] names[i];
  delete [] names;
  delete [] isNodal;

  return 0;
}

void writeMesh(int exoid, int nNodes, int num_elem, int num_elem_block, std::map<int, int> &elem_nodes,
               vector<double> &X, vector<double> &Y, vector<double> &Z, vector< top_block* > &myElements,
               set<int> &deletedElements)
{
  ex_put_init (exoid, "xp2exo", 
	       3, // three dimensions
	       nNodes, // Number of nodes
	       num_elem-deletedElements.size(),
	       num_elem_block,
	       0, 0);

  char* coord_names[] = {"xcoor","ycoor","zcoor"};

  ex_put_coord(exoid, &(X[0]), &(Y[0]), &(Z[0]) );

  ex_put_coord_names(exoid, coord_names);

  std::map<int, const char*> elem_types;
  elem_types[1] = "BEAM";
  elem_types[2] = "QUAD";
  elem_types[4] = "TRIANGLE";
  elem_types[5] = "TETRA";
  elem_types[6] = "BEAM";
  elem_types[66] = "BEAM";
  elem_types[101] = "BEAM";
  elem_types[103] = "QUAD";
  elem_types[106] = "BEAM";
  elem_types[107] = "BEAM";
  elem_types[104] = "TRIANGLE";
  elem_types[108] = "TRIANGLE";
  elem_types[117] = "HEX";
  elem_types[188] = "QUAD";
  elem_types[123] = "TETRA";
  elem_types[124] = "WEDGE";
  elem_types[125] = "TETRA";
  elem_types[145] = "HEX";
  elem_types[146] = "TRIANGLE";
  elem_types[172] = "HEX";
  elem_types[4746] = "QUAD";  
  elem_types[506] = "BEAM";
  elem_types[122] = "BEAM";
  elem_types[121] = "BEAM";
  elem_types[120] = "TRIANGLE";
  elem_types[2120] = "QUAD";
  int i = 0;
  int bc = 0;
  for (i = 0; i < myElements.size(); ++i) {

    top_block& tb = *myElements[i];
    for (top_block::iterator itr = tb.begin(); itr != tb.end(); ++itr) {
      int nn = elem_nodes[itr->second.elid];
      ex_put_elem_block(exoid, bc+1, elem_types[itr->second.elid], 
			itr->second.connect->size()/(nn),
			nn, 0);
    
      ex_put_elem_conn(exoid,bc+1, &((*(itr->second.connect))[0]) );
      ++bc;
    }
  }
}

void writeGlobalVar(int exoid, int numV, char **names, int *decomposition)
{
  ex_put_var_param (exoid, "n", numV);
  ex_put_var_names(exoid, "n", numV, names);
  if (decomposition) {
    char* dec_var_name[] = {"decomposition"};
    ex_put_var_param (exoid, "e", 1);
    ex_put_var_names(exoid, "e", 1, dec_var_name);
  }
}

void deleteElements(int &num_elem, vector< top_block* > &myElements, set<int> &deletedElements,
                    std::map<int, int> &elem_nodes)
{
  // delete elements from the topology element set, and delete faces of deleted elements from the surface topologies
  for (int i = 0; i < 1; ++i) {

    top_block& tb = *myElements[i];
    for (top_block::iterator itr = tb.begin(); itr != tb.end(); ++itr) {
      int nn = elem_nodes[itr->second.elid];
      std::list<int>::iterator itr2 = itr->second.elems.begin();
      std::vector<int>::iterator itr3 = itr->second.connect->begin();
      while(itr2 != itr->second.elems.end()) {
        if(deletedElements.find(*itr2) != deletedElements.end()) {
          std::cerr << "deleting element " << *itr2 << " from " << itr->second.name << std::endl;
#if (__cplusplus >= 201103L) || defined(HACK_INTEL_COMPILER_ITS_CPP11)
          for(int j=1; j<myElements.size(); ++j) {
            top_block& fb = *myElements[j];
            for (top_block::iterator itr4 = fb.begin(); itr4 != fb.end(); ++itr4) {
              if(itr4->second.name.compare(0,8,"surface_") != 0) continue;
              int fnn = elem_nodes[itr4->second.elid];
              std::list<int>::iterator itr5 = itr4->second.elems.begin();
              std::vector<int>::iterator itr6 = itr4->second.connect->begin();
              while(itr5 != itr4->second.elems.end()) {
                if(std::all_of(itr6, itr6+fnn, [&](int k) { return (std::find(itr3, itr3+nn, k) != itr3+nn); })) {
                  std::cerr << "deleting face " << *itr5 << " from " << itr4->second.name << std::endl;
                  itr5 = itr4->second.elems.erase(itr5);
                  itr6 = itr4->second.connect->erase(itr6, itr6+fnn);
                  num_elem--;
                }
                else {
                 itr5++;
                 itr6 += fnn;
                }
              }
            }
          }
#endif
          itr2 = itr->second.elems.erase(itr2);
          itr3 = itr->second.connect->erase(itr3, itr3+nn);
        }
        else {
          itr2++;
          itr3 += nn;
        }
      }
    }
  }
  num_elem -= deletedElements.size();
}

