#include <cmath>

#define one 1.0
#define zero 0.0
#define rdwarf 3.834e-20
#define rgiant 1.304e19

double enorm(int n, double *x)
{
/*
c     **********
c
c     function enorm
c
c     given an n-vector x, this function calculates the
c     euclidean norm of x.
c
c     the euclidean norm is computed by accumulating the sum of
c     squares in three different sums. the sums of squares for the
c     small and large components are scaled so that no overflows
c     occur. non-destructive underflows are permitted. underflows
c     and overflows do not occur in the computation of the unscaled
c     sum of squares for the intermediate components.
c     the definitions of small, intermediate and large components
c     depend on two constants, rdwarf and rgiant. the main
c     restrictions on these constants are that rdwarf**2 not
c     underflow and rgiant**2 not overflow. the constants
c     given here are suitable for every known computer.
c
c     the function statement is
c
c       double precision function enorm(n,x)
c
c     where
c
c       n is a positive integer input variable.
c
c       x is an input array of length n.
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
    double s1 = zero;
    double s2 = zero;
    double s3 = zero;
    double x1max = zero;
    double x3max = zero;

    double agiant = rgiant/(double)n;

    for(int i = 0; i < n; i++) {
        double xabs = fabs(x[i]);
        if (xabs <= rdwarf || xabs >= agiant) {
            if (xabs > rdwarf) {
/*
c
c              sum for large components.
c
*/
                if (xabs > x1max) {
                    s1 = one + s1*(x1max/xabs)*(x1max/xabs);
                    x1max = xabs;
                } else {
                    s1 += (xabs/x1max)*(xabs/x1max);                    
                }
            }
            else {
/*
c
c              sum for small components.
c
*/
                if (xabs > x3max) {
                    s3 = one + s3*(x3max/xabs)*(x3max/xabs);
                    x3max = xabs;
                }
                else {
                    if (xabs != zero) {
                        s3 += (xabs/x3max)*(xabs/x3max);
                    }
                }
            }         
        }
        else {
/*
c
c           sum for intermediate components.
c
*/
            s2 += xabs*xabs;
        }
    }

/*
c
c     calculation of norm.
c
*/
    if (s1 != zero) {
        return x1max*sqrt(s1+(s2/x1max)/x1max);
    }
    else {
        if (s2 != zero) {
            if (s2 >= x3max) return sqrt(s2*(one+(x3max/s2)*(x3max*s3)));
            return sqrt(x3max*((s2/x3max)+(x3max*s3)));
        }
        return x3max*sqrt(s3);
    }
}