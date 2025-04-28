#pragma onece

int lmdif1(
    int (*fcn)(int,int,double *,double *),
    int m,
    int n,
    double *x,
    double *fvec,
    double tol = 1e-10);
