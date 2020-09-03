data
{
    int<lower=0> N;                    // number of wells
    real<lower=0> tmax[N];             // estimated total time
    int<lower=0> d[N];                 // number of divisions before first gfp cell appears
}

parameters
{
    real<lower=0>  q;
}

//transformed parameters
//{
//    real<lower=0>  q;   // the break rate
//    q = 1-q_raw/(q_raw+1);
//
//}

model
{
  for (n in 1:N){
     d[n] ~ neg_binomial_2(1, q);
    }
}
