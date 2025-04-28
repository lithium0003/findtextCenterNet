#include <cmath>

#define p5 5.0e-1
#define p25 2.5e-1
#define zero 0.0

void qrsolv(int n,double *r,int ldr,int *ipvt,double *diag,double *qtb,double *x,double *sdiag)
{
/*
c     **********
c
c     subroutine qrsolv
c
c     given an m by n matrix a, an n by n diagonal matrix d,
c     and an m-vector b, the problem is to determine an x which
c     solves the system
c
c           a*x = b ,     d*x = 0 ,
c
c     in the least squares sense.
c
c     this subroutine completes the solution of the problem
c     if it is provided with the necessary information from the
c     qr factorization, with column pivoting, of a. that is, if
c     a*p = q*r, where p is a permutation matrix, q has orthogonal
c     columns, and r is an upper triangular matrix with diagonal
c     elements of nonincreasing magnitude, then qrsolv expects
c     the full upper triangle of r, the permutation matrix p,
c     and the first n components of (q transpose)*b. the system
c     a*x = b, d*x = 0, is then equivalent to
c
c                  t       t
c           r*z = q *b ,  p *d*p*z = 0 ,
c
c     where x = p*z. if this system does not have full rank,
c     then a least squares solution is obtained. on output qrsolv
c     also provides an upper triangular matrix s such that
c
c            t   t               t
c           p *(a *a + d*d)*p = s *s .
c
c     s is computed within qrsolv and may be of separate interest.
c
c     the subroutine statement is
c
c       subroutine qrsolv(n,r,ldr,ipvt,diag,qtb,x,sdiag,wa)
c
c     where
c
c       n is a positive integer input variable set to the order of r.
c
c       r is an n by n array. on input the full upper triangle
c         must contain the full upper triangle of the matrix r.
c         on output the full upper triangle is unaltered, and the
c         strict lower triangle contains the strict upper triangle
c         (transposed) of the upper triangular matrix s.
c
c       ldr is a positive integer input variable not less than n
c         which specifies the leading dimension of the array r.
c
c       ipvt is an integer input array of length n which defines the
c         permutation matrix p such that a*p = q*r. column j of p
c         is column ipvt(j) of the identity matrix.
c
c       diag is an input array of length n which must contain the
c         diagonal elements of the matrix d.
c
c       qtb is an input array of length n which must contain the first
c         n elements of the vector (q transpose)*b.
c
c       x is an output array of length n which contains the least
c         squares solution of the system a*x = b, d*x = 0.
c
c       sdiag is an output array of length n which contains the
c         diagonal elements of the upper triangular matrix s.
c
c       wa is a work array of length n.
c
c     subprograms called
c
c       fortran-supplied ... dabs,dsqrt
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
*/
    double *wa = new double[n];
/*
c
c     copy r and (q transpose)*b to preserve input and initialize s.
c     in particular, save the diagonal elements of r in x.
c
*/
    for(int j = 0; j < n; j++) {
        for(int i = j; i < n; i++) {
            r[i+j*ldr] = r[j+i*ldr];
        }
        x[j] = r[j+j*ldr];
        wa[j] = qtb[j];
    }
/*
c
c     eliminate the diagonal matrix d using a givens rotation.
c
*/
    for(int j = 0; j < n; j++) {
/*
c
c        prepare the row of d to be eliminated, locating the
c        diagonal element using p from the qr factorization.
c
*/
        if (diag[ipvt[j]] != zero) {
            for(int k = j; k < n; k++) {
                sdiag[k] = zero;
            }
            sdiag[j] = diag[ipvt[j]];
/*
c
c        the transformations to eliminate the row of d
c        modify only a single element of (q transpose)*b
c        beyond the first n, which is initially zero.
c
*/
            double qtbpj = zero;
            for(int k = j; k < n; k++) {
/*
c
c           determine a givens rotation which eliminates the
c           appropriate element in the current row of d.
c
*/
                if (sdiag[k] == zero) continue;
                double cos,cotan,sin,tan;
                if (fabs(r[k+k*ldr]) < fabs(sdiag[k])) {
                    cotan = r[k+k*ldr]/sdiag[k];
                    sin = p5/sqrt(p25+p25*cotan*cotan);
                    cos = sin*cotan;
                }
                else {
                    tan = sdiag[k]/r[k+k*ldr];
                    cos = p5/sqrt(p25+p25*tan*tan);
                    sin = cos*tan;
                }
/*
c
c           compute the modified diagonal element of r and
c           the modified element of ((q transpose)*b,0).
c
*/
                r[k+k*ldr] = cos*r[k+k*ldr] + sin*sdiag[k];
                double temp = cos*wa[k] + sin*qtbpj;
                qtbpj = -sin*wa[k] + cos*qtbpj;
                wa[k] = temp;
/*
c
c           accumulate the tranformation in the row of s.
c
*/
                for(int i = k+1; i < n; i++) {
                    temp = cos*r[i+k*ldr] + sin*sdiag[i];
                    sdiag[i] = -sin*r[i+k*ldr] + cos*sdiag[i];
                    r[i+k*ldr] = temp;
                }
            }
        }
/*
c
c        store the diagonal element of s and restore
c        the corresponding diagonal element of r.
c
*/
        sdiag[j] = r[j+j*ldr];
        r[j+j*ldr] = x[j];
    }
/*
c
c     solve the triangular system for z. if the system is
c     singular, then obtain a least squares solution.
c
*/
    int nsing = n;
    for(int j = 0; j < n; j++) {
        if (sdiag[j] == zero && nsing == n) nsing = j - 1;
        if (nsing < n) wa[j] = zero;
    }
    for(int k = 1; k <= nsing; k++) {
        int j = nsing - k;
        double sum = zero;
        for(int i = j+1; i < nsing; i++) {
            sum += r[i+j*ldr]*wa[i];
        }
        wa[j] = (wa[j] - sum)/sdiag[j];
    }
/*
c
c     permute the components of z back to components of x.
c
*/
    for(int j = 0; j < n; j++) {
        x[ipvt[j]] = wa[j];
    }    

    delete[] wa;
}