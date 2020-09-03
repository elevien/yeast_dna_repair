data
{
    int<lower=0> N;                    // number of wells
    real<lower=0> tmax[N];             // estimated total time
    real<lower=0> t[N];                // time of appearance of first gfp cell
}

parameters
{
    real<lower=0>  beta;   // the break rate
}

model
{
  for (n in 1:N){
     t[n] ~ exponential(beta)[;tmax[n]];
    }
}
