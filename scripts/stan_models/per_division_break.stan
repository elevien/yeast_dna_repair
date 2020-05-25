data
{
    int<lower=0> N;       // number of wells
    vector[N] tgfp;       // times of first gfp observation
}

parameters
{
    real<lower=0>  beta;  // the break rate
}

model
{
    tgfp ~ exponential(beta);
}
