import pandas as pd
import numpy as np
import pystan,pickle



##########################################################################
# inference using stan
##########################################################################


def fill_missing(x):
    K = len(x)
    x_new = [x[0]]
    for j in range(1,K):
        if x[j]>=x_new[-1]:
            x_new.append(x[j])
        else:
            x_new.append(x_new[-1])
    return np.array(x_new)


def constant_break_rate_inference(experiment):
    model = pickle.load(open('./stan_models/constant_break_rate.pkl', 'rb'))
    K = len(experiment['t'][0])
    N = len(experiment['AllCellsBF'][0,:])
    m = experiment['AllCellsBF'].T
    r = experiment['AllCellsGFP'].T
    m2 = np.array([fill_missing(mk) for mk in m])
    r2 = np.array([fill_missing(rk) for rk in r])

    stan_data = {'N':N,
             'K':K,
             'm':m2,
             'r':r2,
             'dt':20}

    fit = model.sampling(data=stan_data, iter=10000, chains=4)
    df = pd.DataFrame(fit.extract())
    filename = './output/constant_break_rate/'+experiment['experiment']+'.csv'
    df.to_csv(filename)








##########################################################################
# functions obtaining jump information
##########################################################################

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
