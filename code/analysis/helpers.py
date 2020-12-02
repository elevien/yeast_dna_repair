import numpy as np
import pandas as pd


def binned(X,Y,bin_size):
    """
    group the X values into bins of size bin_size and
    compute the average and standard deviation of the
    Y values
    """
    Y_sorted = [x for _,x in sorted(zip(X,Y))]
    X_sorted = sorted(X)
    L = len(X)
    X_chunks = [X_sorted[x:x+bin_size] for x in range(0,L-1,bin_size)]
    Y_chunks = [Y_sorted[x:x+bin_size] for x in range(0,L-1,bin_size)]
    Xb = np.array([np.mean(x) for x in X_chunks])
    Yvar = np.array([np.var(y) for y in Y_chunks])
    Yb = np.array([np.mean(y) for y in Y_chunks])
    return Xb,Yb,Yvar


def sigma_GN(df,n_bins):
    X,Y = df.bf.values,df.gfp.values
    bin_ends = np.linspace(X[0],X[-1],n_bins+1)
    # inds = [np.array([k for k in range(len(X)) if X[k]> bin_ends[j] and X[k] < bin_ends[j+1]])\
     #       for j in range(n_bins)]
    inds = [np.where((X > bin_ends[j]) & (X <= bin_ends[j+1])) for j in range(n_bins)]
    Xb = (bin_ends[1:]+bin_ends[:-1])/2
    Y_var = np.array([np.var(Y[i]) for i in inds])
    return Xb,Y_var
