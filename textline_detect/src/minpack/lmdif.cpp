#include <cmath>

#define mcheps 2.2204460492503131e-16

#define one 1.0
#define p1 1.0e-1
#define p5 5.0e-1
#define p25 2.5e-1
#define p75 7.5e-1
#define p0001 1.0e-4
#define zero 0.0

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))

double enorm(int n, double *x);
int fdjac2(int (*fcn)(int,int,double *,double *),int m,int n,double *x,double *fvec,double *fjac,int ldfjac,double epsfcn);
void qrfac(int m,int n,double *a,int lda,bool pivot,int *ipvt,int lipvt,double *rdiag,double *acnorm);
void lmpar(int n,double *r,int ldr,int *ipvt,double *diag,double *qtb,double delta,double &par,double *x,double *sdiag);

int lmdif(
    int (*fcn)(int,int,double *,double *),
    int m,
    int n,
    double *x,
    double *fvec,
    double ftol,
    double xtol,
    double gtol,
    int maxfev,
    double epsfcn,
    double *diag,
    int mode,
    double factor,
    int nprint,
    int &nfev,
    double *fjac,
    int ldfjac,
    int *ipvt,
    double *qtf)
{
/*
c     **********
c
c     subroutine lmdif
c
c     the purpose of lmdif is to minimize the sum of the squares of
c     m nonlinear functions in n variables by a modification of
c     the levenberg-marquardt algorithm. the user must provide a
c     subroutine which calculates the functions. the jacobian is
c     then calculated by a forward-difference approximation.
c
c     the subroutine statement is
c
c       subroutine lmdif(fcn,m,n,x,fvec,ftol,xtol,gtol,maxfev,epsfcn,
c                        diag,mode,factor,nprint,info,nfev,fjac,
c                        ldfjac,ipvt,qtf,wa1,wa2,wa3,wa4)
c
c     where
c
c       fcn is the name of the user-supplied subroutine which
c         calculates the functions. fcn must be declared
c         in an external statement in the user calling
c         program, and should be written as follows.
c
c         subroutine fcn(m,n,x,fvec,iflag)
c         integer m,n,iflag
c         double precision x(n),fvec(m)
c         ----------
c         calculate the functions at x and
c         return this vector in fvec.
c         ----------
c         return
c         end
c
c         the value of iflag should not be changed by fcn unless
c         the user wants to terminate execution of lmdif.
c         in this case set iflag to a negative integer.
c
c       m is a positive integer input variable set to the number
c         of functions.
c
c       n is a positive integer input variable set to the number
c         of variables. n must not exceed m.
c
c       x is an array of length n. on input x must contain
c         an initial estimate of the solution vector. on output x
c         contains the final estimate of the solution vector.
c
c       fvec is an output array of length m which contains
c         the functions evaluated at the output x.
c
c       ftol is a nonnegative input variable. termination
c         occurs when both the actual and predicted relative
c         reductions in the sum of squares are at most ftol.
c         therefore, ftol measures the relative error desired
c         in the sum of squares.
c
c       xtol is a nonnegative input variable. termination
c         occurs when the relative error between two consecutive
c         iterates is at most xtol. therefore, xtol measures the
c         relative error desired in the approximate solution.
c
c       gtol is a nonnegative input variable. termination
c         occurs when the cosine of the angle between fvec and
c         any column of the jacobian is at most gtol in absolute
c         value. therefore, gtol measures the orthogonality
c         desired between the function vector and the columns
c         of the jacobian.
c
c       maxfev is a positive integer input variable. termination
c         occurs when the number of calls to fcn is at least
c         maxfev by the end of an iteration.
c
c       epsfcn is an input variable used in determining a suitable
c         step length for the forward-difference approximation. this
c         approximation assumes that the relative errors in the
c         functions are of the order of epsfcn. if epsfcn is less
c         than the machine precision, it is assumed that the relative
c         errors in the functions are of the order of the machine
c         precision.
c
c       diag is an array of length n. if mode = 1 (see
c         below), diag is internally set. if mode = 2, diag
c         must contain positive entries that serve as
c         multiplicative scale factors for the variables.
c
c       mode is an integer input variable. if mode = 1, the
c         variables will be scaled internally. if mode = 2,
c         the scaling is specified by the input diag. other
c         values of mode are equivalent to mode = 1.
c
c       factor is a positive input variable used in determining the
c         initial step bound. this bound is set to the product of
c         factor and the euclidean norm of diag*x if nonzero, or else
c         to factor itself. in most cases factor should lie in the
c         interval (.1,100.). 100. is a generally recommended value.
c
c       nprint is an integer input variable that enables controlled
c         printing of iterates if it is positive. in this case,
c         fcn is called with iflag = 0 at the beginning of the first
c         iteration and every nprint iterations thereafter and
c         immediately prior to return, with x and fvec available
c         for printing. if nprint is not positive, no special calls
c         of fcn with iflag = 0 are made.
c
c       info is an integer output variable. if the user has
c         terminated execution, info is set to the (negative)
c         value of iflag. see description of fcn. otherwise,
c         info is set as follows.
c
c         info = 0  improper input parameters.
c
c         info = 1  both actual and predicted relative reductions
c                   in the sum of squares are at most ftol.
c
c         info = 2  relative error between two consecutive iterates
c                   is at most xtol.
c
c         info = 3  conditions for info = 1 and info = 2 both hold.
c
c         info = 4  the cosine of the angle between fvec and any
c                   column of the jacobian is at most gtol in
c                   absolute value.
c
c         info = 5  number of calls to fcn has reached or
c                   exceeded maxfev.
c
c         info = 6  ftol is too small. no further reduction in
c                   the sum of squares is possible.
c
c         info = 7  xtol is too small. no further improvement in
c                   the approximate solution x is possible.
c
c         info = 8  gtol is too small. fvec is orthogonal to the
c                   columns of the jacobian to machine precision.
c
c       nfev is an integer output variable set to the number of
c         calls to fcn.
c
c       fjac is an output m by n array. the upper n by n submatrix
c         of fjac contains an upper triangular matrix r with
c         diagonal elements of nonincreasing magnitude such that
c
c                t     t           t
c               p *(jac *jac)*p = r *r,
c
c         where p is a permutation matrix and jac is the final
c         calculated jacobian. column j of p is column ipvt(j)
c         (see below) of the identity matrix. the lower trapezoidal
c         part of fjac contains information generated during
c         the computation of r.
c
c       ldfjac is a positive integer input variable not less than m
c         which specifies the leading dimension of the array fjac.
c
c       ipvt is an integer output array of length n. ipvt
c         defines a permutation matrix p such that jac*p = q*r,
c         where jac is the final calculated jacobian, q is
c         orthogonal (not stored), and r is upper triangular
c         with diagonal elements of nonincreasing magnitude.
c         column j of p is column ipvt(j) of the identity matrix.
c
c       qtf is an output array of length n which contains
c         the first n elements of the vector (q transpose)*fvec.
c
c       wa1, wa2, and wa3 are work arrays of length n.
c
c       wa4 is a work array of length m.
c
c     subprograms called
c
c       user-supplied ...... fcn
c
c       minpack-supplied ... dpmpar,enorm,fdjac2,lmpar,qrfac
c
c       fortran-supplied ... dabs,dmax1,dmin1,dsqrt,mod
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
*/

/*
c
c     epsmch is the machine precision.
c
*/
    double epsmch = mcheps;

    int info = 0;
    nfev = 0;
/*
c
c     check the input parameters for errors.
c
*/
    if(n <= 0 || m < n || ldfjac < m || ftol < zero || xtol < zero || gtol < zero || maxfev <= 0 || factor <= zero) {
        return info;
    }
    if (mode == 2) {
        for(int j = 0; j < n; j++) {
            if (diag[j] <= zero) return info;
        }
    }
/*
c
c     evaluate the function at the starting point
c     and calculate its norm.
c
*/
    nfev = 1;
    if(fcn(m,n,x,fvec) != 0) {
        return -1;
    }
    double fnorm = enorm(m,fvec);

    double *wa1 = new double[n];
    double *wa2 = new double[n];
    double *wa3 = new double[n];
    double *wa4 = new double[m];
/*
c
c     initialize levenberg-marquardt parameter and iteration counter.
c
*/
    double par = zero;
    int iter = 1;
    double delta = 0;
    double xnorm = 0;
    double actred = 0;
/*
c
c     beginning of the outer loop.
c
*/
    while(true) {
/*
c
c        calculate the jacobian matrix.
c
*/
        nfev += n;
        if(fdjac2(fcn,m,n,x,fvec,fjac,ldfjac,epsfcn) < 0) {
            delete[] wa1;
            delete[] wa2;
            delete[] wa3;
            delete[] wa4;
            return -1;
        }
/*
c
c        compute the qr factorization of the jacobian.
c
*/
        qrfac(m,n,fjac,ldfjac,true,ipvt,n,wa1,wa2);
/*
c
c        on the first iteration and if mode is 1, scale according
c        to the norms of the columns of the initial jacobian.
c
*/
        if (iter == 1) {
            if (mode != 2) {
                for(int j = 0; j < n; j++) {
                    diag[j] = wa2[j] == zero ? one : wa2[j];
                }
            }
/*
c
c        on the first iteration, calculate the norm of the scaled x
c        and initialize the step bound delta.
c
*/
            for(int j = 0; j < n; j++) {
                wa3[j] = diag[j]*x[j];
            }
            xnorm = enorm(n,wa3);
            delta = factor*xnorm;
            if (delta == zero) delta = factor;
        }
/*
c
c        form (q transpose)*fvec and store the first n components in
c        qtf.
c
*/
        for(int i = 0; i < m; i++) {
            wa4[i] = fvec[i];
        }
        for(int j = 0; j < n; j++) {
            if (fjac[j+j*ldfjac] != zero) {
                double sum = zero;
                for(int i = j; i < m; i++) {
                    sum += fjac[i+j*ldfjac]*wa4[i];
                }
                double temp = -sum/fjac[j+j*ldfjac];
                for(int i = j; i < m; i++) {
                    wa4[i] += fjac[i+j*ldfjac]*temp;
                }
            }
            fjac[j+j*ldfjac] = wa1[j];
            qtf[j] = wa4[j];
        }
/*
c
c        compute the norm of the scaled gradient.
c
*/
        double gnorm = zero;
        if (fnorm != zero) {
            for(int j = 0; j < n; j++) {
                if (wa2[ipvt[j]] == zero) continue;
                double sum = zero;
                for(int i = 0; i <= j; i++) {
                    sum += fjac[i+j*ldfjac]*(qtf[i]/fnorm);
                }
                gnorm = MAX(gnorm,fabs(sum/wa2[ipvt[j]]));
            }
        }
/*
c
c        test for convergence of the gradient norm.
c
*/
        if (gnorm <= gtol) info = 4;
        if (info != 0) {
            delete[] wa1;
            delete[] wa2;
            delete[] wa3;
            delete[] wa4;
            return info;
        }

/*
c
c        rescale if necessary.
c
*/
        if (mode != 2) {
            for(int j = 0; j < n; j++) {
                diag[j] = MAX(diag[j],wa2[j]);
            }
        }
/*
c
c        beginning of the inner loop.
c
*/
        double ratio = zero;
        while(ratio < p0001) {
/*
c
c           determine the levenberg-marquardt parameter.
c
*/
            lmpar(n,fjac,ldfjac,ipvt,diag,qtf,delta,par,wa1,wa2);
/*
c
c           store the direction p and x + p. calculate the norm of p.
c
*/
            for(int j = 0; j < n; j++) {
                wa1[j] = -wa1[j];
                wa2[j] = x[j] + wa1[j];
                wa3[j] = diag[j]*wa1[j];
            }
            double pnorm = enorm(n,wa3);
/*
c
c           on the first iteration, adjust the initial step bound.
c
*/
            if (iter == 1) delta = MIN(delta,pnorm);
/*
c
c           evaluate the function at x + p and calculate its norm.
c
*/
            nfev++;
            if(fcn(m,n,wa2,wa4) < 0) {
                delete[] wa1;
                delete[] wa2;
                delete[] wa3;
                delete[] wa4;
                return -1;                    
            }
            double fnorm1 = enorm(m,wa4);
/*
c
c           compute the scaled actual reduction.
c
*/
            actred = -one;
            if (p1*fnorm1 < fnorm) actred = one - (fnorm1/fnorm)*(fnorm1/fnorm);
/*
c
c           compute the scaled predicted reduction and
c           the scaled directional derivative.
c
*/
            for(int j = 0; j < n; j++) {
                wa3[j] = zero;
                double temp = wa1[ipvt[j]];
                for(int i = 0; i <= j; i++) {
                    wa3[i] += fjac[i+j*ldfjac]*temp;
                }
            }
            double temp1 = enorm(n,wa3)/fnorm;
            double temp2 = (sqrt(par)*pnorm)/fnorm;
            double prered = temp1*temp1 + temp2*temp2/p5;
            double dirder = -(temp1*temp1 + temp2*temp2);
/*
c
c           compute the ratio of the actual to the predicted
c           reduction.
c
*/
            ratio = zero;
            if (prered != zero) ratio = actred/prered;
/*
c
c           update the step bound.
c
*/
            if (ratio <= p25) {
                double temp;
                if (actred >= zero){
                    temp = p5;
                }
                else {
                    temp = p5*dirder/(dirder + p5*actred);
                }
                if (p1*fnorm1 >= fnorm || temp < p1) temp = p1;
                delta = temp*MIN(delta,pnorm/p1);
                par = par/temp;
            }
            else {
                if (par == zero || ratio >= p75) {
                    delta = pnorm/p5;
                    par = p5*par;
                }
            }
/*
c
c           test for successful iteration.
c
*/
            if (ratio >= p0001) {
/*
c
c           successful iteration. update x, fvec, and their norms.
c
*/
                for(int j = 0; j < n; j++) {
                    x[j] = wa2[j];
                    wa2[j] = diag[j]*x[j];
                }
                for(int i = 0; i < m; i++) {
                    fvec[i] = wa4[i];
                }
                xnorm = enorm(n,wa2);
                fnorm = fnorm1;
                iter++;
            }
/*
c
c           tests for convergence.
c
*/
            if (fabs(actred) <= ftol && prered <= ftol && p5*ratio <= one) info = 1;
            if (delta <= xtol*xnorm) info = 2;
            if (fabs(actred) <= ftol && prered <= ftol && p5*ratio <= one && info == 2) info = 3;
            if (info != 0) {
                delete[] wa1;
                delete[] wa2;
                delete[] wa3;
                delete[] wa4;
                return info;
            }
/*
c
c           tests for termination and stringent tolerances.
c
*/
            if (nfev >= maxfev) info = 5;
            if (fabs(actred) <= epsmch && prered <= epsmch && p5*ratio <= one) info = 6;
            if (delta <= epsmch*xnorm) info = 7;
            if (gnorm <= epsmch) info = 8;
            if (info != 0) {
                delete[] wa1;
                delete[] wa2;
                delete[] wa3;
                delete[] wa4;
                return info;
            }
/*
c
c           end of the inner loop. repeat if iteration unsuccessful.
c
*/
        }
/*
c
c        end of the outer loop.
c
*/   
    }
}

int lmdif1(
    int (*fcn)(int,int,double *,double *),
    int m,
    int n,
    double *x,
    double *fvec,
    double tol)
{
/*
c     **********
c
c     subroutine lmdif1
c
c     the purpose of lmdif1 is to minimize the sum of the squares of
c     m nonlinear functions in n variables by a modification of the
c     levenberg-marquardt algorithm. this is done by using the more
c     general least-squares solver lmdif. the user must provide a
c     subroutine which calculates the functions. the jacobian is
c     then calculated by a forward-difference approximation.
c
c     the subroutine statement is
c
c       subroutine lmdif1(fcn,m,n,x,fvec,tol,info,iwa,wa,lwa)
c
c     where
c
c       fcn is the name of the user-supplied subroutine which
c         calculates the functions. fcn must be declared
c         in an external statement in the user calling
c         program, and should be written as follows.
c
c         subroutine fcn(m,n,x,fvec,iflag)
c         integer m,n,iflag
c         double precision x(n),fvec(m)
c         ----------
c         calculate the functions at x and
c         return this vector in fvec.
c         ----------
c         return
c         end
c
c         the value of iflag should not be changed by fcn unless
c         the user wants to terminate execution of lmdif1.
c         in this case set iflag to a negative integer.
c
c       m is a positive integer input variable set to the number
c         of functions.
c
c       n is a positive integer input variable set to the number
c         of variables. n must not exceed m.
c
c       x is an array of length n. on input x must contain
c         an initial estimate of the solution vector. on output x
c         contains the final estimate of the solution vector.
c
c       fvec is an output array of length m which contains
c         the functions evaluated at the output x.
c
c       tol is a nonnegative input variable. termination occurs
c         when the algorithm estimates either that the relative
c         error in the sum of squares is at most tol or that
c         the relative error between x and the solution is at
c         most tol.
c
c       info is an integer output variable. if the user has
c         terminated execution, info is set to the (negative)
c         value of iflag. see description of fcn. otherwise,
c         info is set as follows.
c
c         info = 0  improper input parameters.
c
c         info = 1  algorithm estimates that the relative error
c                   in the sum of squares is at most tol.
c
c         info = 2  algorithm estimates that the relative error
c                   between x and the solution is at most tol.
c
c         info = 3  conditions for info = 1 and info = 2 both hold.
c
c         info = 4  fvec is orthogonal to the columns of the
c                   jacobian to machine precision.
c
c         info = 5  number of calls to fcn has reached or
c                   exceeded 200*(n+1).
c
c         info = 6  tol is too small. no further reduction in
c                   the sum of squares is possible.
c
c         info = 7  tol is too small. no further improvement in
c                   the approximate solution x is possible.
c
c       iwa is an integer work array of length n.
c
c       wa is a work array of length lwa.
c
c       lwa is a positive integer input variable not less than
c         m*n+5*n+m.
c
c     subprograms called
c
c       user-supplied ...... fcn
c
c       minpack-supplied ... lmdif
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
*/
    if (n <= 0 || m < n || tol < zero) return 0;
    int maxfev = 200*(n + 1);
    double ftol = tol;
    double xtol = tol;
    double gtol = zero;
    double epsfcn = zero;
    int mode = 1;
    int nprint = 0;
    int nfev = 0;
    double factor = 1.0e2;
    double *wa = new double[m*n+n*2];
    int *iwa = new int[n];
    int info = lmdif(fcn,m,n,x,fvec,ftol,xtol,gtol,
        maxfev,epsfcn,wa,mode,factor,nprint,nfev,&wa[n*2],m,iwa,
        &wa[n]);
    delete[] wa;
    delete[] iwa;
    return info;
}
