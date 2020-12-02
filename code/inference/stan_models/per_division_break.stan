data
{
    int<lower=0> N;                    // number of wells
    int<lower=0> L;                    // number of time points
    real<lower=0> t[L];                // time points
    vector[L] N[N];                 // number of cells
    vector[L] Ng[N];                 // number of green cells
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
  for (n in 1:N){// go through each well
     d[n] ~ neg_binomial_2(1, q);
    }
}
