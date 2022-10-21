#ifndef _VECTOR_H_
#define _VECTOR_H_

#include <iostream>
#include <complex>
using std::cerr;
using std::endl;

using std::complex;

//------------------------------------------------------------------------------

template<class A, class B>
class ProdRes {
  public:
    typedef B ResType;
};


template<>
class ProdRes<complex<double>, double> {
  public:
    typedef complex<double> ResType;
};

//------------------------------------------------------------------------------

template<class T, class Scalar> 
class Expr {

public:

  int len;
  T x;

  Expr(T v) : x(v) { len = x.size(); }
  Expr(T v, int l) : x(v) { len = l; }

  Scalar operator[] (int i) const { return x[i]; }
  int size() const { return len; }

};

//------------------------------------------------------------------------------

inline
double sqNorm(double x) { return x*x;}

template <class Scalar>
inline
double sqNorm(complex<Scalar> x) { return (x*conj(x)).real(); }

template<class Scalar> inline Scalar DotTerm(const Scalar &a, const Scalar &b) {return a*b; }

template<class Scalar>
complex<Scalar> DotTerm(const complex<Scalar> &a, const complex<Scalar> &b)
{
	          return conj(b)*a;
}

//------------------------------------------------------------------------------

template<class Scalar>
class Vec {

public:

  typedef int InfoType;

  int len;
  bool locAlloc;
  Scalar *v;

  Vec(int, Scalar * = 0);
  Vec(const Vec<Scalar> &);

  template<class T>
  Vec(const Expr<T, Scalar> &);

  ~Vec() { if (locAlloc && v) delete [] v; }

  template<class T>
  Vec<Scalar> &operator=(const T);

  template<class T>
  Vec<Scalar> &operator*=(const T);

  Vec<Scalar> &operator=(const Vec<Scalar> &);
  Vec<Scalar> &operator+=(const Vec<Scalar> &);
  Vec<Scalar> &operator-=(const Vec<Scalar> &);

  Scalar operator*(const Vec<Scalar> &);

  template<class T>
  Vec<Scalar> &operator=(const Expr<T, Scalar> &);

  template<class T>
  Vec<Scalar> &operator+=(const Expr<T, Scalar> &);

  template<class T>
  Vec<Scalar> &operator-=(const Expr<T, Scalar> &);

  template<class T>
  Scalar operator*(const Expr<T, Scalar> &);

  Scalar &operator[](int i) const { return v[i]; }

  int size() const { return len; }

  double sizeMB() const { return static_cast<double>(len * sizeof(Scalar)) / (1024.0*1024.0); }

  Scalar *data() const { return v; }

/*
void resize(int l) {
  int minlen = (l < len) ? l : len;
  Scalar* tmp = 0;
  if (v) {
    if (len > 0) {
      tmp = new Scalar[len];
      for (int i=0; i<minlen; ++i) tmp[i] = v[i];
    }
    delete [] v;
  }
  v = new Scalar[l];
  if (tmp) {
    for (int i=0; i<minlen; ++i) v[i] = tmp[i];
    delete [] tmp;
  }
  locAlloc = true;
  len = l;
} 
*/

  void resize(int l) 
  { 
    Scalar* tmp = 0;
    if (v) {
      if (len > 0) {
	tmp = new Scalar[len];
	for (int i=0; i<len; ++i)
	  tmp[i] = v[i];
      }
      delete [] v; 
    }
    v = new Scalar[l];
    if (tmp) {
      int tmplen; 
      if (len < l)
        tmplen = len;
      else
        tmplen = l;
      for (int i=0; i<tmplen; ++i)
	v[i] = tmp[i];
      delete [] tmp;
    }
    locAlloc = true;
    len = l; 
  }

  Scalar min() const {
    Scalar vmin = v[0];
    for (int i=1; i<len; ++i) 
      vmin = v[i] < vmin ? v[i] : vmin;
    return vmin;
  }

  Scalar max() const {
    Scalar vmax = v[0];
    for (int i=1; i<len; ++i) 
      vmax = v[i] > vmax ? v[i] : vmax;
    return vmax;
  }

  Scalar sum() const { 
    Scalar res = 0; 
    for (int i=0; i<len; ++i) 
      res += v[i]; 
    return res;
  }

  double norm() const {
    double res = 0;
    for (int i=0; i<len; ++i)
    res += sqNorm(v[i]);
    return sqrt(res);
  }

#ifdef USE_IOSTREAM
  void print(char *msg = "") const {
    cerr.flush();
    if (msg) cerr << msg << endl;
    for (int i=0; i<len; ++i) 
      cerr << i << ": "<< v[i] << endl; 
    cerr.flush();
  }
#endif

};

//------------------------------------------------------------------------------

template<class Scalar>
inline
Vec<Scalar>::Vec(int l, Scalar *vv) 
{ 

  len = l;

  if (vv) {
    locAlloc = false;
    v = vv;
  }
  else {
    locAlloc = true;
    if (len > 0)
      v = new Scalar[len];
    else
      v = 0;
  }

}

//------------------------------------------------------------------------------

template<class Scalar>
inline
Vec<Scalar>::Vec(const Vec<Scalar> &y) 
{

  len = y.len; 
  locAlloc = true;

  v = new Scalar[len]; 

  for (int i=0; i<len; ++i) v[i] = y.v[i];

}
//------------------------------------------------------------------------------

template<class Scalar>
template<class T>
inline
Vec<Scalar>::Vec(const Expr<T, Scalar> &expr) 
{

  len = expr.len; 
  locAlloc = true;

  v = new Scalar[len]; 

  const T &x = expr.x;
  for (int i=0; i<len; ++i) v[i] = x[i];

}

//------------------------------------------------------------------------------

template<class Scalar>
template<class T>
inline
Vec<Scalar> &
Vec<Scalar>::operator=(const T y)
{

  for (int i=len; i--; ) v[i] = y;

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar>
template<class T>
inline
Vec<Scalar> &
Vec<Scalar>::operator*=(const T y)
{

  for (int i=len; i--; ) v[i] *= y;

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar>
inline
Vec<Scalar> &
Vec<Scalar>::operator=(const Vec<Scalar> &y)
{

  for (int i=len; i--; ) v[i] = y.v[i];

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar>
inline
Vec<Scalar> &
Vec<Scalar>::operator+=(const Vec<Scalar> &y)
{

  for (int i=len; i--; ) v[i] += y.v[i];

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar>
inline
Vec<Scalar> &
Vec<Scalar>::operator-=(const Vec<Scalar> &y)
{

  for (int i=len; i--; ) v[i] -= y.v[i];

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar>
inline
Scalar
Vec<Scalar>::operator*(const Vec<Scalar> &y) 
{

  Scalar res = 0;

  for (int i=0; i<len; ++i) res += v[i] * y.v[i];

  return res;

}

//------------------------------------------------------------------------------

template<class Scalar>
template<class T>
inline
Vec<Scalar> &
Vec<Scalar>::operator=(const Expr<T, Scalar> &expr)
{

  const T &x = expr.x;

  for (int i=len; i--; ) v[i] = x[i];

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar>
template<class T>
inline
Vec<Scalar> &
Vec<Scalar>::operator+=(const Expr<T, Scalar> &expr)
{

  const T &x = expr.x;

  for (int i=len; i--; ) v[i] += x[i];

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar>
template<class T>
inline
Vec<Scalar> &
Vec<Scalar>::operator-=(const Expr<T, Scalar> &expr)
{

  const T &x = expr.x;

  for (int i=len; i--; ) v[i] -= x[i];

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar>
template<class T>
inline
Scalar 
Vec<Scalar>::operator*(const Expr<T, Scalar> &expr)
{

  const T &x = expr.x;

  Scalar res = 0;

  for (int i=0; i<len; ++i) res += v[i] * x[i];

  return res;

}

//------------------------------------------------------------------------------

template<class T1, class T2, class Scalar>
class Sum {

  T1 a;
  T2 b;
  int len;

public:

  Sum(T1 aa, T2 bb, int l) : a(aa), b(bb) { len = l; }

  Scalar operator[](int i) const { return a[i]+b[i]; }
  int size() const { return len; }

};

//------------------------------------------------------------------------------

template<class T1, class T2, class Scalar>
inline
Expr<Sum<T1, T2, Scalar>, Scalar>
operator+(const Expr<T1, Scalar> &x1, const Expr<T2, Scalar> &x2)
{

  return Expr<Sum<T1, T2, Scalar>, Scalar> 
    ( Sum<T1, T2, Scalar>(x1.x, x2.x, x1.size()) );

}

//------------------------------------------------------------------------------

template<class Scalar>
inline
Expr<Sum<Scalar *, Scalar *, Scalar>, Scalar>
operator+(const Vec<Scalar> &v1, const Vec<Scalar> &v2)
{

  return Expr<Sum<Scalar *, Scalar *, Scalar>, Scalar>
    ( Sum<Scalar *, Scalar *, Scalar>(v1.v, v2.v, v1.size()) );

}

//------------------------------------------------------------------------------

template<class T, class Scalar>
inline
Expr<Sum<T, Scalar *, Scalar>, Scalar>
operator+(const Expr<T, Scalar> &x, const Vec<Scalar> &v)
{

  return Expr<Sum<T, Scalar *, Scalar>, Scalar>
    ( Sum<T, Scalar *, Scalar>(x.x, v.v, v.size()) );

}

//------------------------------------------------------------------------------

template<class T, class Scalar>
inline
Expr<Sum<Scalar *, T, Scalar>, Scalar>
operator+(const Vec<Scalar> &v, const Expr<T, Scalar> &x)
{

  return Expr<Sum<Scalar *, T, Scalar>, Scalar>
    ( Sum<Scalar *, T, Scalar>(v.v, x.x, v.size()) );

}

//------------------------------------------------------------------------------

template<class T1, class T2, class Scalar>
class Diff {

  T1 a;
  T2 b;
  int len;

public:

  Diff(T1 aa, T2 bb, int l) : a(aa), b(bb) { len = l; }

  Scalar operator[](int i) const { return a[i]-b[i]; }
  int size() const { return len; }

};

//------------------------------------------------------------------------------

template<class T1, class T2, class Scalar>
inline
Expr<Diff<T1, T2, Scalar>, Scalar>
operator-(const Expr<T1, Scalar> &x1, const Expr<T2, Scalar> &x2)
{

  return Expr<Diff<T1, T2, Scalar>, Scalar>
    ( Diff<T1, T2, Scalar>(x1.x, x2.x, x1.size()) );

}

//------------------------------------------------------------------------------

template<class Scalar>
inline
Expr<Diff<Scalar *, Scalar *, Scalar>, Scalar>
operator-(const Vec<Scalar> &v1, const Vec<Scalar> &v2)
{

  return Expr<Diff<Scalar *, Scalar *, Scalar>, Scalar>
    ( Diff<Scalar *, Scalar *, Scalar>(v1.v, v2.v, v1.size()) );

}

//------------------------------------------------------------------------------

template<class T, class Scalar>
inline
Expr<Diff<T, Scalar *, Scalar>, Scalar>
operator-(const Expr<T, Scalar> &x, const Vec<Scalar> &v)
{

  return Expr<Diff<T, Scalar *, Scalar>, Scalar>
    ( Diff<T, Scalar *, Scalar>(x.x, v.v, v.size()) );

}

//------------------------------------------------------------------------------

template<class T, class Scalar>
inline
Expr<Diff<Scalar *, T, Scalar>, Scalar>
operator-(const Vec<Scalar> &v, const Expr<T, Scalar> &x)
{

  return Expr<Diff<Scalar *, T, Scalar>, Scalar>
    ( Diff<Scalar *, T, Scalar>(v.v, x.x, v.size()) );

}

//------------------------------------------------------------------------------

template<class T1, class T2, class Scalar>
class Div {

  T1 a;
  T2 b;
  int len;

public:

  Div(T1 aa, T2 bb, int l) : a(aa), b(bb) { len = l; }

  Scalar operator[](int i) const { return a[i]/b[i]; }
  int size() const { return len; }

};

//------------------------------------------------------------------------------

template<class T1, class T2, class Scalar>
inline
Expr<Div<T1, T2, Scalar>, Scalar>
operator/(const Expr<T1, Scalar> &x1, const Expr<T2, Scalar> &x2)
{

  return Expr<Div<T1, T2, Scalar>, Scalar>
    ( Div<T1, T2, Scalar>(x1.x, x2.x, x1.size()) );

}

//------------------------------------------------------------------------------

template<class Scalar>
inline
Expr<Div<Scalar *, Scalar *, Scalar>, Scalar>
operator/(const Vec<Scalar> &v1, const Vec<Scalar> &v2)
{

  return Expr<Div<Scalar *, Scalar *, Scalar>, Scalar>
    ( Div<Scalar *, Scalar *, Scalar>(v1.v, v2.v, v1.size()) );

}

//------------------------------------------------------------------------------

template<class T, class Scalar>
inline
Expr<Div<T, Scalar *, Scalar>, Scalar>
operator/(const Expr<T, Scalar> &x, const Vec<Scalar> &v)
{

  return Expr<Div<T, Scalar *, Scalar>, Scalar>
    ( Div<T, Scalar *, Scalar>(x.x, v.v, v.size()) );

}

//------------------------------------------------------------------------------

template<class T, class Scalar>
inline
Expr<Div<Scalar *, T, Scalar>, Scalar>
operator/(const Vec<Scalar> &v, const Expr<T, Scalar> &x)
{

  return Expr<Div<Scalar *, T, Scalar>, Scalar>
    ( Div<Scalar *, T, Scalar>(v.v, x.x, v.size()) );

}

//------------------------------------------------------------------------------

template<class T, class Scalar, class Res>
class OuterProd {

  Scalar y;
  T a;
  int len;

public:

  OuterProd(Scalar yy, T aa, int l) : y(yy), a(aa) { len = l; }

  Res operator[](int i) const { return y*a[i]; }
  int size() const { return len; }

};

//------------------------------------------------------------------------------

template<class T, class S2>
inline
Expr<OuterProd<T, double, typename ProdRes<double, S2>::ResType>, 
    typename ProdRes<double,S2>::ResType > operator*(double y, const Expr<T, S2> &x)
{

  return Expr<OuterProd<T, double, typename ProdRes<double,S2>::ResType>, 
         typename ProdRes<double,S2>::ResType>
    ( OuterProd<T, double, typename ProdRes<double,S2>::ResType>(y, x.x, x.size()) );
}

//------------------------------------------------------------------------------

template<class T, class S2>
inline
Expr<OuterProd<T, complex<double>, typename ProdRes<complex<double>, S2>::ResType>, 
    typename ProdRes<complex<double>,S2>::ResType > operator*(complex<double> y, const Expr<T, S2> &x)
{

  return Expr<OuterProd<T, complex<double>, typename ProdRes<complex<double>,S2>::ResType>, 
         typename ProdRes<complex<double>,S2>::ResType>
    ( OuterProd<T, complex<double>, typename ProdRes<complex<double>,S2>::ResType>(y, x.x, x.size()) );
}

//------------------------------------------------------------------------------
template<class Scalar, class Res>
inline
Expr<OuterProd<Res *, Scalar, typename ProdRes<Scalar,Res>::ResType>, 
     typename ProdRes<Scalar,Res>::ResType> operator*(Scalar y, const Vec<Res> &v)
{

  return Expr<OuterProd<Res *, Scalar, typename ProdRes<Scalar,Res>::ResType>, 
         typename ProdRes<Scalar,Res>::ResType> 
    ( OuterProd<Res *, Scalar, typename ProdRes<Scalar,Res>::ResType>(y, v.v, v.size()) );

}

//------------------------------------------------------------------------------

template<class T1, class T2, class Scalar>
class InnerProd {

  T1 a;
  T2 b;
  int len;

public:

  InnerProd(T1 aa, T2 bb, int l) : a(aa), b(bb) { len = l; }

  Scalar operator[](int i) const { return a[i]*b[i]; }
  int size() const { return len; }

};

//------------------------------------------------------------------------------
/*
template<class T1, class T2, class Scalar>
inline
Scalar
operator*(const Expr<T1, Scalar> &x, const Expr<T2, Scalar> &y)
{

  const T1 &xx = x.x;
  const T2 &yy = y.x;

  Scalar res = 0;

  for (int i=0; i<x.size(); ++i) 
    res += xx[i] * yy[i];

  return res;

}

//------------------------------------------------------------------------------

template<class T1, class Scalar>
inline
Scalar
operator*(const Expr<T1, Scalar> &x, const Vec<Scalar> &v)
{

  const T1 &xx = x.x;

  Scalar res = 0;

  for (int i=0; i<x.size(); ++i) 
    res += v[i] * xx[i];

  return res;

}
*/
//------------------------------------------------------------------------------

template<class Scalar, int dim>
class SVec { 

public:

  typedef int InfoType;

  int len;
  bool locAlloc;
  Scalar (*v)[dim];

  SVec(int, Scalar (*)[dim] = 0);
  SVec(const SVec<Scalar,dim> &);

  template<class T>
  SVec(const Expr<T, Scalar> &);

  ~SVec() { if (locAlloc && v) delete [] v; }
     
  SVec<Scalar,dim> &operator=(const Scalar);
  SVec<Scalar,dim> &operator*=(const Scalar);

  SVec<Scalar,dim> &operator=(const SVec<Scalar,dim> &);
  SVec<Scalar,dim> &operator+=(const SVec<Scalar,dim> &);
  SVec<Scalar,dim> &operator-=(const SVec<Scalar,dim> &);

  Scalar operator*(const SVec<Scalar,dim> &);

  template<class T>
  SVec<Scalar,dim> &operator=(const Expr<T, Scalar> &);

  template<class T>
  SVec<Scalar,dim> &operator+=(const Expr<T, Scalar> &);

  template<class T>
  SVec<Scalar,dim> &operator-=(const Expr<T, Scalar> &);

  template<class T>
  Scalar operator*(const Expr<T, Scalar> &);

  void set(const Scalar *);

  Scalar *operator[] (int i) const { return v[i]; }

  int size() const { return len; }

  double sizeMB() const { return static_cast<double>(len*dim * sizeof(Scalar)) / (1024.0*1024.0); }

  Scalar (*data() const)[dim] { return v; }

  double norm()  {
    double res = 0.0;
    for (int i = 0; i < len; i++)
      for (int k = 0; k < dim; k++)
        res += sqNorm(v[i][k]);
    return sqrt(res);
  }

  double* sum()  {
    double *res = new double[dim];
		for (int k=0; k<dim; k++) res[k]=0.0;
    for (int i = 0; i < len; i++)
      for (int k = 0; k < dim; k++)
        res[k] += v[i][k];
    return res;
  }

  void resize(int l) 
  {
   int minlen = (l < len) ? l : len;
   Scalar (*tmp)[dim] = 0;
   if (v) {
    if (len > 0) {
      tmp = new Scalar[len][dim];
      for (int i=0; i<minlen; ++i) 
         for (int j=0; j<dim; ++j)
              tmp[i][j] = v[i][j];
    }
    delete [] v;
   }
   v = new Scalar[l][dim];
   if (tmp) {
    for (int i=0; i<minlen; ++i) 
       for (int j=0; j<dim; ++j)
          v[i][j] = tmp[i][j];
    delete [] tmp;
   }
   locAlloc = true;
   len = l;
  }

/*
  void resize(int l) 
  { 
    Scalar (*tmp)[dim] = 0;
    if (v) {
      if (len > 0) {
	tmp = new Scalar[len][dim];
	for (int i=0; i<len; ++i)
	  for (int j=0; j<dim; ++j)
	    tmp[i][j] = v[i][j];
      }
      delete [] v; 
    }
    v = new Scalar[l][dim];
    if (tmp) {
      for (int i=0; i<l; ++i)
	for (int j=0; j<dim; ++j)
	  v[i][j] = tmp[i][j];
      delete [] tmp;
    }
    locAlloc = true;
    len = l; 
  }
*/

  Scalar min() const {
    Scalar vmin = v[0][0];
    for (int i=0; i<len; ++i)
      for (int j=0; j<dim; ++j)
        vmin = v[i][j] < vmin ? v[i][j] : vmin;
    return vmin;
  }

  Scalar max() const {
    Scalar vmax = v[0][0];
    for (int i=0; i<len; ++i)
      for (int j=0; j<dim; ++j)
        vmax = v[i][j] > vmax ? v[i][j] : vmax;
    return vmax;
  }

#ifdef USE_IOSTREAM
  void print(char *msg = "") { 
    if (msg) cerr << msg << endl;
    for (int i=0; i<len; ++i) {
      cerr << i << ": ";
      for (int j=0; j<dim; ++j) 
	cerr << v[i][j] << " ";
      cerr << endl; 
      cerr.flush();
    }
  }
#endif

};

//------------------------------------------------------------------------------

template<class Scalar, int dim>
inline
SVec<Scalar,dim>::SVec(int l, Scalar (*vv)[dim]) 
{ 

  len = l;

  if (vv) {
    locAlloc = false; 
    v = vv;
  }
  else {
    locAlloc = true;
    if (len > 0) 
      v = new Scalar[len][dim]; 
    else 
      v = 0;
  }

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
inline
SVec<Scalar,dim>::SVec(const SVec<Scalar,dim> &y) 
{

  len = y.len;
  locAlloc = true;

  v = new Scalar[len][dim];

  Scalar *vv = reinterpret_cast<Scalar *>(v);
  const Scalar *yy = reinterpret_cast<Scalar *>(y.v);

  for (int i=0; i<dim*len; ++i) vv[i] = yy[i];

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
template<class T>
inline
SVec<Scalar,dim>::SVec(const Expr<T, Scalar> &expr) 
{

  len = expr.len;
  locAlloc = true;

  v = new Scalar[len][dim];

  Scalar *vv = reinterpret_cast<Scalar *>(v);
  const T &x = expr.x;

  for (int i=0; i<dim*len; ++i) vv[i] = x[i];

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
inline
void
SVec<Scalar,dim>::set(const Scalar *y)
{

  for (int i=0; i<len; ++i) 
    for (int j=0; j<dim; ++j)
      v[i][j] = y[j];

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
inline
SVec<Scalar,dim> &
SVec<Scalar,dim>::operator=(const Scalar y)
{

  Scalar *vv = reinterpret_cast<Scalar *>(v);

  for (int i=0; i<dim*len; ++i) vv[i] = y;

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
inline
SVec<Scalar,dim> &
SVec<Scalar,dim>::operator*=(const Scalar y)
{

  Scalar *vv = reinterpret_cast<Scalar *>(v);

  for (int i=0; i<dim*len; ++i) vv[i] *= y;

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
inline
SVec<Scalar,dim> &
SVec<Scalar,dim>::operator=(const SVec<Scalar,dim> &y)
{

  const Scalar *yy = reinterpret_cast<Scalar *>(y.v);
  Scalar *vv = reinterpret_cast<Scalar *>(v);

  for (int i=0; i<dim*len; ++i) vv[i] = yy[i];

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
inline
SVec<Scalar,dim> &
SVec<Scalar,dim>::operator+=(const SVec<Scalar,dim> &y)
{

  const Scalar *yy = reinterpret_cast<Scalar *>(y.v);
  Scalar *vv = reinterpret_cast<Scalar *>(v);

  for (int i=0; i<dim*len; ++i) vv[i] += yy[i];

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
inline
SVec<Scalar,dim> &
SVec<Scalar,dim>::operator-=(const SVec<Scalar,dim> &y)
{

  const Scalar *yy = reinterpret_cast<Scalar *>(y.v);
  Scalar *vv = reinterpret_cast<Scalar *>(v);

  for (int i=0; i<dim*len; ++i) vv[i] -= yy[i];

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
inline
Scalar 
SVec<Scalar,dim>::operator*(const SVec<Scalar,dim> &y) 
{

  Scalar res = 0;

  const Scalar *vv = reinterpret_cast<Scalar *>(v);
  const Scalar *yy = reinterpret_cast<Scalar *>(y.v);

  for (int i=0; i<dim*len; ++i) res += vv[i]*yy[i];

  return res;

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
template<class T>
inline
SVec<Scalar,dim> &
SVec<Scalar,dim>::operator=(const Expr<T, Scalar> &expr)
{

  const T &x = expr.x;
  Scalar *vv = reinterpret_cast<Scalar *>(v);

  for (int i=0; i<dim*len; ++i) vv[i] = x[i];

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
template<class T>
inline
SVec<Scalar,dim> &
SVec<Scalar,dim>::operator+=(const Expr<T, Scalar> &expr)
{

  const T &x = expr.x;
  Scalar *vv = reinterpret_cast<Scalar *>(v);

  for (int i=0; i<dim*len; ++i) vv[i] += x[i];

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
template<class T>
inline
SVec<Scalar,dim> &
SVec<Scalar,dim>::operator-=(const Expr<T, Scalar> &expr)
{

  const T &x = expr.x;
  Scalar *vv = reinterpret_cast<Scalar *>(v);

  for (int i=0; i<dim*len; ++i) vv[i] -= x[i];

  return *this;

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
template<class T>
inline
Scalar 
SVec<Scalar,dim>::operator*(const Expr<T, Scalar> &expr) 
{

  Scalar res = 0;

  const T &x = expr.x;
  const Scalar *vv = reinterpret_cast<Scalar *>(v);

  for (int i=0; i<dim*len; ++i) res += vv[i]*x[i];

  return res;

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
inline
Expr<Sum<Scalar *, Scalar *, Scalar>, Scalar>
operator+(const SVec<Scalar,dim> &v1, const SVec<Scalar,dim> &v2)
{

  return Expr<Sum<Scalar *, Scalar *, Scalar>, Scalar>
    ( Sum<Scalar *, Scalar *, Scalar>(reinterpret_cast<Scalar *>(v1.v), 
				      reinterpret_cast<Scalar *>(v2.v), v1.size()*dim) );

}

//------------------------------------------------------------------------------

template<class T, class Scalar, int dim>
inline
Expr<Sum<T, Scalar *, Scalar>, Scalar>
operator+(const Expr<T, Scalar> &x, const SVec<Scalar,dim> &v)
{

  return Expr<Sum<T, Scalar *, Scalar>, Scalar>
    ( Sum<T, Scalar *, Scalar>(x.x, reinterpret_cast<Scalar *>(v.v), x.size()) );

}

//------------------------------------------------------------------------------

template<class T, class Scalar, int dim>
inline
Expr<Sum<Scalar *, T, Scalar>, Scalar>
operator+(const SVec<Scalar,dim> &v, const Expr<T, Scalar> &x)
{

  return Expr<Sum<Scalar *, T, Scalar>, Scalar>
    ( Sum<Scalar *, T, Scalar>(reinterpret_cast<Scalar *>(v.v), x.x, x.size()) );

}

//------------------------------------------------------------------------------

template<class Scalar, int dim>
inline
Expr<Diff<Scalar *, Scalar *, Scalar>, Scalar>
operator-(const SVec<Scalar,dim> &v1, const SVec<Scalar,dim> &v2)
{

  return Expr<Diff<Scalar *, Scalar *, Scalar>, Scalar>
    ( Diff<Scalar *, Scalar *, Scalar>(reinterpret_cast<Scalar *>(v1.v), 
				       reinterpret_cast<Scalar *>(v2.v), v1.size()*dim) );

}

//------------------------------------------------------------------------------

template<class T, class Scalar, int dim>
inline
Expr<Diff<T, Scalar *, Scalar>, Scalar>
operator-(const Expr<T, Scalar> &x, const SVec<Scalar,dim> &v)
{

  return Expr<Diff<T, Scalar *, Scalar>, Scalar>
    ( Diff<T, Scalar *, Scalar>(x.x, reinterpret_cast<Scalar *>(v.v), x.size()) );

}

//------------------------------------------------------------------------------

template<class T, class Scalar, int dim>
inline
Expr<Diff<Scalar *, T, Scalar>, Scalar>
operator-(const SVec<Scalar,dim> &v, const Expr<T, Scalar> &x)
{

  return Expr<Diff<Scalar *, T, Scalar>, Scalar>
    ( Diff<Scalar *, T, Scalar>(reinterpret_cast<Scalar *>(v.v), x.x, x.size()) );

}

//------------------------------------------------------------------------------

template<class Scalar, class Res, int dim>
inline
Expr<OuterProd<Res *, Scalar, typename ProdRes<Scalar,Res>::ResType>, typename ProdRes<Scalar,Res>::ResType>
operator*(Scalar y, const SVec<Res,dim> &v)
{

  return Expr<OuterProd<Res *, Scalar, typename ProdRes<Scalar,Res>::ResType>, 
         typename ProdRes<Scalar,Res>::ResType>
    ( OuterProd<Res *, Scalar, typename ProdRes<Scalar,Res>::ResType>(y, reinterpret_cast<Res *>(v.v), v.size()*dim) );

}

//------------------------------------------------------------------------------
/*
template<class T1, class Scalar, int dim>
inline
Scalar
operator*(const Expr<T1, Scalar> &x, const SVec<Scalar,dim> &v)
{

  const T1 &xx = x.x;
  Scalar *vv = reinterpret_cast<Scalar *>(v.v);

  Scalar res = 0;

  for (int i=0; i<x.size(); ++i) 
    res += vv[i] * xx[i];

  return res;

}
*/
//------------------------------------------------------------------------------

#endif
