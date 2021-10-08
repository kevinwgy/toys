#include<fstream>
#include<iostream>
#include<algorithm>
#include<map>
#include<set>
using namespace std;

struct edge {
  int n1, n2;
  edge() {n1 = n2 = -1;}
  edge(int _n1, int _n2) {if(_n1<=_n2) {n1=_n1;n2=_n2;} else {n1=_n2;n2=_n1;}}
  ~edge() {}
  bool operator<(const edge & e2) const {
    if(this->n1<e2.n1)
      return true;
    else if (this->n1>e2.n1)
      return false;
    else if (this->n2<e2.n2)
      return true;
    else return false;
    return false;
  }
  edge &operator=(const edge &e2) {
    this->n1 = e2.n1;
    this->n2 = e2.n2;
    return *this;
  }
};

struct triangle {
  int n[3];
  triangle() {n[0] = n[1] = n[2] = -1;}
  triangle(int _n1, int _n2, int _n3) {n[0] = _n1; n[1] = _n2; n[2] = _n3; sort(n,n+3);} 
  ~triangle() {}
  bool operator<(const triangle & e2) const {
    if(this->n[0]<e2.n[0])
      return true;
    else if(this->n[0]>e2.n[0])
      return false;
    else if(this->n[1]<e2.n[1])
      return true;
    else if(this->n[1]>e2.n[1])
      return false;
    else if(this->n[2]<e2.n[2])
      return true;
    else if(this->n[2]>e2.n[2])
      return false;
    else return false;
    return false;
  }
  triangle &operator=(const triangle &e2) {
    this->n[0] = e2.n[0];
    this->n[1] = e2.n[1];
    this->n[2] = e2.n[2];
  }
};

int main(int argc, char* argv[])
{
  set<triangle> triangles;
  set<triangle>::iterator it;

  ifstream infile(argv[1],ios::in);
  int id, a,b,c, code, count;
  id = -1; count = 0;
  char word1[100];
  triangle tria;

  while(!infile.eof()) {
    word1[0] = '0';
    word1[1] = '\n';
    infile >> word1;
    id = strtol(word1,NULL,10);
    if(id==0) {break;}
    count++;
    infile >> code >> a >> b >> c;
    tria = triangle(a,b,c);
    it = triangles.find(tria);
    if(it==triangles.end())
      triangles.insert(tria);
    else
      cout << "Duplicated element! " << id << " " << code << " " << a << " " << b << " " << c << endl;
  }
    
  cout << "Loaded " << count << " elements." << endl;
  infile.close();
  return 0;
}
