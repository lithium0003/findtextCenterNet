#include <cmath>

#define p1 1.0e-1
#define p001 1.0e-3
#define zero 0.0
#define minmag 2.2250738585072014e-308

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

double enorm(int n, double *x);
void qrsolv(int n,double *r,int ldr,int *ipvt,double *diag,double *qtb,double *x,double *sdiag);

void lmpar(int n,double *r,int ldr,int *ipvt,double *diag,double *qtb,double delta,double &par,double *x,double *sdiag)
{
/*
c     **********
c
c     subroutine lmpar
c
c     given an m by n matrix a, an n by n nonsingular diagonal
c     matrix d, an m-vector b, and a positive number delta,
c     the problem is to determine a value for the parameter
c     par such that if x solves the system
c
c           a*x = b ,     sqrt(par)*d*x = 0 ,
c
c     in the least squares sense, and dxnorm is the euclidean
c     norm of d*x, then either par is zero and
c
c           (dxnorm-delta) .le. 0.1*delta ,
c
c     or par is positive and
c
c           abs(dxnorm-delta) .le. 0.1*delta .
c
c     this subroutine completes the solution of the problem
c     if it is provided with the necessary information from the
c     qr factorization, with column pivoting, of a. that is, if
c     a*p = q*r, where p is a permutation matrix, q has orthogonal
c     columns, and r is an upper triangular matrix with diagonal
c     elements of nonincreasing magnitude, then lmpar expects
c     the full upper triangle of r, the permutation matrix p,
c     and the first n components of (q transpose)*b. on output
c     lmpar also provides an upper triangular matrix s such that
c
c            t   t                   t
c           p *(a *a + par*d*d)*p = s *s .
c
c     s is employed within lmpar and may be of separate interest.
c
c     only a few iterations are generally needed for convergence
c     of the algorithm. if, however, the limit of 10 iterations
c     is reached, then the output par will contain the best
c     value obtained so far.
c
c     the subroutine statement is
c
c       subroutine lmpar(n,r,ldr,ipvt,diag,qtb,delta,par,x,sdiag,
c                        wa1,wa2)
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
c       delta is a positive input variable which specifies an upper
c         bound on the euclidean norm of d*x.
c
c       par is a nonnegative variable. on input par contains an
c         initial estimate of the levenberg-marquardt parameter.
c         on output par contains the final estimate.
c
c       x is an output array of length n which contains the least
c         squares solution of the system a*x = b, sqrt(par)*d*x = 0,
c         for the output par.
c
c       sdiag is an output array of length n which contains the
c         diagonal elements of the upper triangular matrix s.
c
c       wa1 and wa2 are work arrays of length n.
c
c     subprograms called
c
c       minpack-supplied ... dpmpar,enorm,qrsolv
c
c       fortran-supplied ... dabs,dmax1,dmin1,dsqrt
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
*/
    double *wa1 = new double[n];
    double *wa2 = new double[n];
/*
c
c     dwarf is the smallest positive magnitude.
c
*/
    double dwarf = minmag;
/*
c
c     compute and store in x the gauss-newton direction. if the
c     jacobian is rank-deficient, obtain a least squares solution.
c
*/
    int nsing = n;
    for(int j = 0; j < n; j++) {
        wa1[j] = qtb[j];
        if (r[j+j*ldr] == zero && nsing == n) nsing = j - 1;
        if (nsing < n) wa1[j] = zero;
    }
    for(int k = 1; k <= nsing; k++) {
        int j = nsing - k;
        wa1[j] /= r[j+j*ldr];

        double temp = wa1[j];
        for(int i = 0; i < j; i++) {
            wa1[i] -= r[i+j*ldr]*temp;
        }
    }
    for(int j = 0; j < n; j++) {
        x[ipvt[j]] = wa1[j];
    }
/*
c
c     initialize the iteration counter.
c     evaluate the function at the origin, and test
c     for acceptance of the gauss-newton direction.
c
*/
    int iter = 0;
    for(int j = 0; j < n; j++) {
        wa2[j] = diag[j]*x[j];
    }
    double dxnorm = enorm(n,wa2);
    double fp = dxnorm - delta;
    if (fp <= p1*delta) {
/*
c
c     termination.
c
*/        
        if (iter == 0) par = zero;
        delete[] wa1;
        delete[] wa2;
        return;
    }

/*
c
c     if the jacobian is not rank deficient, the newton
c     step provides a lower bound, parl, for the zero of
c     the function. otherwise set this bound to zero.
c
*/
    double parl = zero;
    if (nsing >= n) {
        for(int j = 0; j < n; j++) {
            wa1[j] = diag[ipvt[j]]*(wa2[ipvt[j]]/dxnorm);
        }
        for(int j = 0; j < n; j++) {
            double sum = zero;
            for(int i = 0; i < j; i++) {
                sum += r[i+j*ldr]*wa1[i];
            }
            wa1[j] = (wa1[j] - sum)/r[j+j*ldr];
        }
        double temp = enorm(n,wa1);
        parl = ((fp/delta)/temp)/temp;
    }
/*
c
c     calculate an upper bound, paru, for the zero of the function.
c
*/
    for(int j = 0; j < n; j++) {
        double sum = zero;
        for(int i = 0; i <= j; i++) {
            sum += r[i+j*ldr]*qtb[i];
        }
        wa1[j] = sum/diag[ipvt[j]];
    }
    double gnorm = enorm(n,wa1);
    double paru = gnorm/delta;
    if (paru == zero) paru = dwarf/MIN(delta,p1);
/*
c
c     if the input par lies outside of the interval (parl,paru),
c     set par to the closer endpoint.
c
*/
    par = MAX(par,parl);
    par = MIN(par,paru);
    if (par == zero) par = gnorm/dxnorm;
/*
c
c     beginning of an iteration.
c
*/
    while(true) {
        iter++;
/*
c
c        evaluate the function at the current value of par.
c
*/
        if (par == zero) par = MAX(dwarf,p001*paru);
        double temp = sqrt(par);
        for(int j = 0; j < n; j++) {
            wa1[j] = temp*diag[j];
        }
        qrsolv(n,r,ldr,ipvt,wa1,qtb,x,sdiag);
        for(int j = 0; j < n; j++) {
            wa2[j] = diag[j]*x[j];
        }
        dxnorm = enorm(n,wa2);
        temp = fp;
        fp = dxnorm - delta;
/*
c
c        if the function is small enough, accept the current value
c        of par. also test for the exceptional cases where parl
c        is zero or the number of iterations has reached 10.
c
*/
        if(fabs(fp) <= p1*delta || (parl == zero && fp <= temp && temp < zero) || iter == 10) {
            delete[] wa1;
            delete[] wa2;
            return;
        }
/*
c
c        compute the newton correction.
c
*/ 
        for(int j = 0; j < n; j++) {
            wa1[j] = diag[ipvt[j]]*(wa2[ipvt[j]]/dxnorm);
        }
        for(int j = 0; j < n; j++) {
            wa1[j] /= sdiag[j];
            temp = wa1[j];
            for(int i = j+1; i < n; i++) {
                wa1[i] -= r[i+j*ldr]*temp;
            }
        }
        temp = enorm(n,wa1);
        double parc = ((fp/delta)/temp)/temp;
/*
c
c        depending on the sign of the function, update parl or paru.
c
*/
        if (fp > zero) parl = MAX(parl,par);
        if (fp < zero) paru = MIN(paru,par);
/*
c
c        compute an improved estimate for par.
c
*/
        par = MAX(parl,par+parc);
/*
c
c        end of an iteration.
c
*/
    }
}