import random,copy,math,time,os,csv,sys
import numpy as np
from scipy import optimize

def normal_func(x,mu,sigma):
    return 1/(sigma*np.sqrt(np.pi*2))*np.exp(-(x-mu)**2/(2*sigma**2))

def normal_min_func(x,mu,sigma):
    z = (x-mu)/sigma
    return (np.sqrt(2/np.pi)*(1 + (-1 - sci.special.erf(z/np.sqrt(2)))/2.))/np.exp(z**2/2.)/sigma

def get_passage_times(cells,t,n_crit):
    """
    get the first passage times to reach a critical size

    Input:
        cells  -  an array of trajectories. cells[:,k] is the kth trajectory
        t      -  the list of times. must have len(t) = len(cells[:,k])
        n_crit -  the critical value of n

    Output:
        lags   - a list of times for each trajectory to go beyond n_crit

    """
    lag_inds = [np.argmax(x>n_crit) for x in cells.T]
    return  np.array([t[j] for j in lag_inds])


def get_jumps(t,m,r):
    """
    get information about jump times (times when the number of cells changes)

    Input:
        t      -  the list of times
        m      -  number of modified cells
        r      -  number of repaired cells

    Output:
        T   - the total "cell-cycle time" until the appearence of first gfp cell
        jump_times

    """
    ind_gfp = np.argmax(r>0)
    k = 1
    i_gfp =0 # index of gfp in jump_cells
    jump_times = []
    jump_cells = []
    while k< len(m):
        dm = m[k]-m[k-1]
        if dm>0:
            jump_times.append(t[k])
            jump_cells.append(m[k])
            if k <= ind_gfp:
                i_gfp+=1
        k+=1

    jump_times.append(t[-1])
    jump_times = np.array(jump_times)
    jump_cells = np.array(jump_cells)
    taus = jump_times[1:]-jump_times[:-1]
    if i_gfp>0:
        T = np.sum(jump_cells[:i_gfp]*taus[:i_gfp])
    else:
        T = 0.
    return T,jump_times[:-1],jump_cells[:-1],taus,i_gfp

def get_single_cell_beta(exp):
    t = exp['t'][0]
    T_max = 2000
    Ts = [get_jumps(t,m,r)[0] for m,r in zip(exp['OneCellBF'].T,exp['OneCellGFP'].T) if np.max(r)>0]
    T_avg = np.mean(Ts)
    def obj_func(beta):
        return T_avg - 1/beta #+ np.exp(-T_max*beta)*T_max/(1-np.exp(-T_max*beta))
    res =  optimize.minimize(obj_func, 1/T_avg,tol=1e-20)
    return res.x
