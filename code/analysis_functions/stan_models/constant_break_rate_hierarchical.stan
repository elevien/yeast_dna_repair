data
{
    int<lower=0> M           // total number of observations
    int<lower=0> J           // number of experiments
    int<lower=0> j [M]        // experiments
    int<lower=0> n[M];       // wells
    int<lower=0> k[M];       // time step
    real<lower=0> dt;        // time step (in minutes, assumed the same for all experiments)
    int r[M];       // number of gfp cells
    int m[M];       // number bright field cells
}

parameters
{
    real<lower=0>  beta[J];   // the break rate
    real<lower=0>  alpha[J];  // the division rate
}

model
{
  real mu1;
  real mu2;
  int dm;
  int dr;
  for (m in 1:M){
    mu1 = alpha*m[n]*dt;
    mu2 = alpha*r[n,k-1]*dt + beta*(m[n,k-1]-r[n,k-1])*dt;
    dm = m[n,k]-m[n,k-1];
    dr = r[n,k]-r[n,k-1];
    dm ~ poisson (mu1);
    dr ~ poisson (mu2);

    }
}
