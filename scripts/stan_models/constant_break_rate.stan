data
{
    int<lower=0> N;       // number of wells
    int<lower=0> K;       // number of observations
    real<lower=0> dt;     // time step (in minutes)
    int r[N,K];       // number of gfp cells
    int m[N,K];       // number bright field cells
}

parameters
{
    real<lower=0>  beta;   // the break rate
    real<lower=0>  alpha;  // the division rate
}

model
{
  real mu1;
  real mu2;
  int dm;
  int dr;
  for (n in 1:N){
    for (k in 2:K){
      mu1 = alpha*m[n,k-1]*dt;
      mu2 = alpha*r[n,k-1]*dt + beta*(m[n,k-1]-r[n,k-1])*dt;
      dm = m[n,k]-m[n,k-1];
      dr = r[n,k]-r[n,k-1];
      dm ~ poisson (mu1);
      dr ~ poisson (mu2);
      }
    }
}
