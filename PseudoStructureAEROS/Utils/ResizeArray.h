#ifndef _RESIZE_ARRAY_H_
#define _RESIZE_ARRAY_H_

template <class Type> class ResizeArray {
   Type *d;
   Type init_v;
   int  csize;  // current size
   bool myData;
   ResizeArray(const ResizeArray &); // should never be used
   ResizeArray(ResizeArray &); // should never be used
 public:
   ResizeArray(Type i, int ini_size=16);
   ~ResizeArray();
   Type &operator[] (int i);
   void resize(int ns);
   int max_size() { return csize; }
   Type *operator + (int i); //dangerous operator to use with caution.
    // It will not check that the returned pointer is not used beyond
    // the existing values
   Type *yield(); // When we want to keep the array and remove the resizing
   void deleteArray()  { delete [] d; d = 0; csize = 1; }
   void restartArray() { d = new Type[csize]; };
   Type *data(bool _myData = true) { myData = _myData; return d; }
};

template <class Type>
inline
Type &ResizeArray<Type>::operator[](int i) {
 if(i >= csize)
    resize(i);
 return d[i];
}

template <class Type>
inline
Type *ResizeArray<Type>::operator+(int i) {
  if(i >= csize)
    resize(i);
 return d+i;
}


template <class Type>
ResizeArray<Type>::ResizeArray(Type iv, int is) 
{
  init_v = iv;
  csize  = is;
  if(csize > 0) {
    d = new Type[csize];
    int i;
    for(i=0; i < csize; ++i)
      d[i] = init_v;
  }
  else d = 0;

  myData = true;
}

template <class Type>
void
ResizeArray<Type>::resize(int ns)
{
  int nsize = (3*csize)/2;
  if(ns >= nsize) nsize=ns+1;
  Type *nd = new Type[nsize];
  int j;
  for(j=0; j < csize; ++j)
     nd[j] = d[j];
  for(; j <nsize; ++j)
    nd[j] = init_v;
  if(d) delete [] d;
  csize=nsize;
  d=nd;
}

template <class Type>
ResizeArray<Type>::~ResizeArray()
{
  if(d && myData) { delete [] d; d = 0; }
}

template  <class Type>
Type *
ResizeArray<Type>::yield()
{
  Type *old = d;
  d = (Type *)0;
  return old;
}
#endif
