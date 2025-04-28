#include <cmath>
#include <utility>

#define mcheps 2.2204460492503131e-16
#define one 1.0
#define p05 5.0e-2
#define zero 0.0

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

double enorm(int n, double *x);

void qrfac(int m,int n,double *a,int lda,bool pivot,int *ipvt,int lipvt,double *rdiag,double *acnorm)
{
/*
c     **********
c
c     subroutine qrfac
c
c     this subroutine uses householder transformations with column
c     pivoting (optional) to compute a qr factorization of the
c     m by n matrix a. that is, qrfac determines an orthogonal
c     matrix q, a permutation matrix p, and an upper trapezoidal
c     matrix r with diagonal elements of nonincreasing magnitude,
c     such that a*p = q*r. the householder transformation for
c     column k, k = 1,2,...,min(m,n), is of the form
c
c                           t
c           i - (1/u(k))*u*u
c
c     where u has zeros in the first k-1 positions. the form of
c     this transformation and the method of pivoting first
c     appeared in the corresponding linpack subroutine.
c
c     the subroutine statement is
c
c       subroutine qrfac(m,n,a,lda,pivot,ipvt,lipvt,rdiag,acnorm,wa)
c
c     where
c
c       m is a positive integer input variable set to the number
c         of rows of a.
c
c       n is a positive integer input variable set to the number
c         of columns of a.
c
c       a is an m by n array. on input a contains the matrix for
c         which the qr factorization is to be computed. on output
c         the strict upper trapezoidal part of a contains the strict
c         upper trapezoidal part of r, and the lower trapezoidal
c         part of a contains a factored form of q (the non-trivial
c         elements of the u vectors described above).
c
c       lda is a positive integer input variable not less than m
c         which specifies the leading dimension of the array a.
c
c       pivot is a logical input variable. if pivot is set true,
c         then column pivoting is enforced. if pivot is set false,
c         then no column pivoting is done.
c
c       ipvt is an integer output array of length lipvt. ipvt
c         defines the permutation matrix p such that a*p = q*r.
c         column j of p is column ipvt(j) of the identity matrix.
c         if pivot is false, ipvt is not referenced.
c
c       lipvt is a positive integer input variable. if pivot is false,
c         then lipvt may be as small as 1. if pivot is true, then
c         lipvt must be at least n.
c
c       rdiag is an output array of length n which contains the
c         diagonal elements of r.
c
c       acnorm is an output array of length n which contains the
c         norms of the corresponding columns of the input matrix a.
c         if this information is not needed, then acnorm can coincide
c         with rdiag.
c
c       wa is a work array of length n. if pivot is false, then wa
c         can coincide with rdiag.
c
c     subprograms called
c
c       minpack-supplied ... dpmpar,enorm
c
c       fortran-supplied ... dmax1,dsqrt,min0
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
*/
    double *wa = new double[n];
/*
c
c     epsmch is the machine precision.
c
*/
    double epsmch = mcheps;
/*
c
c     compute the initial column norms and initialize several arrays.
c
*/    
    for(int j = 0; j < n; j++) {
        acnorm[j] = enorm(m, &a[j*lda]);
        rdiag[j] = acnorm[j];
        wa[j] = rdiag[j];
        if (pivot) ipvt[j] = j;
    }
/*
c
c     reduce a to r with householder transformations.
c
*/
    int minmn = MIN(m,n);
    for(int j = 0; j < minmn; j++) {
        if(pivot) {
/*
c
c        bring the column of largest norm into the pivot position.
c
*/
            int kmax = j;
            for(int k = j; k < n; k++) {
                if (rdiag[k] > rdiag[kmax]) kmax = k;
            }
            if (kmax != j) {
                for(int i = 0; i < m; i++) {
                    std::swap(a[i+j*lda], a[i+kmax*lda]);
                }
                rdiag[kmax] = rdiag[j];
                wa[kmax] = wa[j];
                std::swap(ipvt[j], ipvt[kmax]);
            }
        }
/*
c
c        compute the householder transformation to reduce the
c        j-th column of a to a multiple of the j-th unit vector.
c
*/
        double ajnorm = enorm(m-j, &a[j+j*lda]);
        if (ajnorm != zero) {
            if (a[j+j*lda] < zero) ajnorm = -ajnorm;
            for(int i = j; i < m; i++) {
                a[i+j*lda] /= ajnorm;
            }
            a[j+j*lda] += one;
/*
c
c        apply the transformation to the remaining columns
c        and update the norms.
c
*/
            for(int k = j+1; k < n; k++) {
                double sum = zero;
                for(int i = j; i < m; i++) {
                    sum += a[i+j*lda]*a[i+k*lda];
                }
                double temp = sum/a[j+j*lda];
                for(int i = j; i < m; i++) {
                    a[i+k*lda] -= temp*a[i+j*lda];
                }
                if (pivot && rdiag[k] != zero) {
                    temp = a[j+k*lda]/rdiag[k];
                    rdiag[k] *= sqrt(MAX(zero,one-temp*temp));
                    if(p05*(rdiag[k]/wa[k])*(rdiag[k]/wa[k]) <= epsmch) {
                        rdiag[k] = enorm(m-j-1,&a[j+1+k*lda]);
                        wa[k] = rdiag[k];
                    }
                }
            }
        }
        rdiag[j] = -ajnorm;
    }

    delete[] wa;
}