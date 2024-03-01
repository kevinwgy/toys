/************************************************************************
 * Copyright © 2020 The Multiphysics Modeling and Computation (M2C) Lab
 * <kevin.wgy@gmail.com> <kevinw3@vt.edu>
 ************************************************************************/

#pragma once
#include "Vector3D.h"
#include <cassert>
#include <cfloat> //DBL_MAX

/**************************************************************************
 * This file declares some basic geometry tools
 *************************************************************************/
namespace GeoTools {

const double INTERSECTIONS_EPSILON = 1.0e-14; //this should be a tolerance smaller than "half_thickness"

/**************************************************************************
 * Check if a point in 2D is within a disk
 *   Inputs:
 *     x,y          -- coords of the point
 *     cen_x, cen_y -- coords of the center of the disk
 *     r            -- radius of the disk
 *   Outputs: true or false
 */
inline bool IsPointInDisk(double x, double y, double cen_x, double cen_y, double r)
{
  return ( (x-cen_x)*(x-cen_x) + (y-cen_y)*(y-cen_y) <= r*r);
}

/**************************************************************************
 * Check if a point in 2D is within a rectangle
 *   Inputs:
 *     x,y          -- coords of the point
 *     cen_x, cen_y -- coords of the center of the rectangle
 *     lx, ly       -- dimensions of the rectangle in x and y directions
 *   Outputs: true or false
 */
inline bool IsPointInRectangle(double x, double y, double cen_x, double cen_y, double lx, double ly)
{
  return x >= cen_x - 0.5*lx  &&  x <= cen_x + 0.5*lx  &&  y >= cen_y - 0.5*ly  &&  y <= cen_y + 0.5*ly;
}

/**************************************************************************
 * Project a point onto a line in 3D, specified using an edge (line segment) 
 *   Inputs:
 *     x0 -- coords of the point
 *     xA -- coords of the start point of the edge
 *     xB -- coords of the end point of the edge
 *   Outputs:
 *     alpha -- affine coordinate (i.e. xA + alpha*AB = projection point)
 *     return value -- distance from the point to the line
 *   Note: If the "edge" is actually a point (i.e. xA = xB), alpha will be 0,
 *         and the distance will be the distance to that point
 */
inline double ProjectPointToLine(Vec3D& x0, Vec3D& xA, Vec3D& xB, double &alpha)
{
  Vec3D AB= xB-xA;
  Vec3D AX = x0-xA;
  double length2 = AB*AB;

  alpha = (length2 != 0) ? AB*AX/length2 : 0.0;
  Vec3D P = xA + alpha*AB;
  return (P-x0).norm();
}

/**************************************************************************
 * Calculate the shortest distance from a point to a line segement in 3D 
 *   (this is unsigned distance, i.e. always positive)
 *   Inputs:
 *     x0 -- coords of the point
 *     xA -- coords of the start point of the edge
 *     xB -- coords of the end point of the edge
 *   Outputs:
 *     alpha -- affine coordinate of the closest point (between 0 and 1)
 *     return value -- shortest distance from x0 to the line segment AB
 *   Note: This function can handle the degenerate case of a point (i.e.
 *         xA = xB)
 */
inline double GetShortestDistanceFromPointToLineSegment(Vec3D& x0, Vec3D& xA, Vec3D& xB,
                                                        double &alpha)
{
  double dist = ProjectPointToLine(x0, xA, xB, alpha);
  if(alpha>1.0) {
    dist = (x0-xB).norm();
    alpha = 1.0; 
  } else if (alpha<0.0 || !std::isfinite(alpha)/*xA=xB*/) {
    dist = (x0-xA).norm();
    alpha = 0.0;
  }
  return dist;
}

/**************************************************************************
 * Calculate the normal direction and area of a triangle (by cross product)
 *   Inputs:
 *     xA, xB, xC -- coords of the three vertices of the triangle
 *                    (the order matters!)
 *   Outputs:
 *     dir -- unit normal dir (xB-xA)^(xC-xA)
 *     return value -- area of the triangle
 */
inline double GetNormalAndAreaOfTriangle(Vec3D& xA, Vec3D& xB, Vec3D& xC, 
                                         Vec3D& dir)
{
  Vec3D ABC = 0.5*(xB-xA)^(xC-xA); //cross product
  double area = ABC.norm();
  assert(area != 0.0);
  dir = 1.0/area*ABC;
  return area;
}                                  

/**************************************************************************
 * Project a point onto a plane defined by a point on the plane and the
 * normal direction
 *   Inputs:
 *     x0 -- the point
 *     O  -- a point on the plane
 *     dir -- normal direction
 *     normalized -- (T/F) whether "dir" is normalized (i.e. norm = 1)
 *   Outputs:
 *     return value -- SIGNED distance from the point to the plane, along "dir"
 */
inline double ProjectPointToPlane(Vec3D& x0, Vec3D& O, Vec3D& dir, bool normalized = false)
{
  if(normalized)
    return (x0-O)*dir;

  double norm = dir.norm(); 
  assert(norm!=0.0);
  return (x0-O)*dir/norm;
}


bool
LineSegmentIntersectsPlane(Vec3D X0, Vec3D X1, //!< vertices of line segment
                           Vec3D V0, Vec3D dir, //!< a point on the plane, and its normal
                           double* d, //!< optional output: dist from X0 to intersection
                           Vec3D* xp, //!< optional output: intersection point
                           bool N_normalized) //!< input: whether dir is normalized
{
  if(!N_normalized) {
    double Nnorm = dir.norm();
    assert(Nnorm>0.0);
    dir /= Nnorm;
  }

  Vec3D X01 = X1 - X0;
  double denom = X01*dir;

  if(denom==0) {
    if(d)  *d  = DBL_MAX;
    if(xp) *xp = DBL_MAX;
    return false;    // This line segment is parallel to the plane
  }

  double d0 = ProjectPointToPlane(X0, V0, dir, true);
  double d1 = ProjectPointToPlane(X1, V0, dir, true);

  if(d0*d1<0) {
    if(d || xp) {
      double alpha = fabs(d0)/(fabs(d0)+fabs(d1));
      if(d)
        *d = alpha*X01.norm();
      if(xp)
        *xp = X0 + alpha*X01;
    }
    return true;
  }
  else if(fabs(d0)<INTERSECTIONS_EPSILON) {
    if(d)
      *d = 0.0;
    if(xp)
      *xp = X0;
    return true;
  }
  else if(fabs(d1)<INTERSECTIONS_EPSILON) {
    if(d)
      *d = X01.norm();
    if(xp)
      *xp = X1;
    return true;
  }
  else {//on the same side
    if(d)
      *d = DBL_MAX;
    if(xp)
      *xp = DBL_MAX;
    return false;
  }

  return false; //will never get here
}



/**************************************************************************
 * For a given vector, find two unit vectors such that the three form an
 * orthonormal basis, satisfying the right hand rule (U0-U1-U2). (If the
 * given vector is NOT normalized, this function does not change it.
 * The VALUE of U0 is passed in, not a reference.)
 *   Inputs:
 *     U0 -- a given vector
 *     U0_normalized -- (T/F) whether U0 is normalized.
 *   Outputs:
 *     U1, U2: two unit vectors that are orthogonal to each other, and to U0.
 */
void GetOrthonormalVectors(Vec3D U0, Vec3D &U1, Vec3D &U2, bool U0_normalized)
{

  if(!U0_normalized) {
    double norm = U0.norm();
    assert(norm != 0.0);
    U0 /= norm;
  }

  U1 = U2 = 0.0;
  bool done = false;
  for(int i=0; i<3; i++) {
    if(U0[i]==0) {
      U1[i] = 1.0; //got U1
      bool gotU2 = false;
      for(int j=i+1; j<3; j++) {
        if(U0[j]==0) {
          U2[j] = 1.0; //got U2;
          gotU2 = true;
          break;
        }
      }
      if(!gotU2) {
        int i1 = (i+1) % 3;
        int i2 = (i+2) % 3;
        U2[i1] = -U0[i2];
        U2[i2] = U0[i1];
        U2 /= U2.norm();
      }
      done = true;
      break;
    }
  }
  if(!done) { //!< all the three components of U0 are nonzero
    U1[0] = 1.0;
    U1[1] = 0.0;
    U1[2] = -U0[0]/U0[2];
    U1 /= U1.norm();
    U2[0] = 1.0;
    U2[1] = -(U0[2]*U0[2]/U0[0] + U0[0])/U0[1];
    U2[2] = U0[2]/U0[0];
    U2 /= U2.norm();
  }

  if((U0^U1)*U2<0.0) {//swap U1 and U2 to satisfy the right-hand rule
    Vec3D Utmp = U1;
    U1 = U2;
    U2 = Utmp;
  }

}


} //end of namespace
