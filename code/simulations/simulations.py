import copy,math
import numpy as np


class gillespie_model:
    """

    S : numpy array
        contains the stiochemtric coefficients of each reaction
    rates : function
        takes as input the state y and return the reaction rates

    """
    def __init__(self,S,rates):
        self.S = S
        self.rates = rates


    def gillespie(self,tmax,y_init,max_steps=10**10,*,\
            stop_cond=lambda x: False,dt_sample=0.4):
        """"
        Perform gillespie simulation specified by the model

        Parameters
        ----------
        tmax : float
            maximum time to run simulation
        y_init : array of floats
            the initial state of the system

        Returns
        -------
        t: array
            the time points at which reactions occured
        y: array
            the state of the system at reaction events
        """
        y = np.zeros((max_steps,len(y_init)))
        y[0] = y_init
        t = np.zeros(max_steps)
        t_cur = 0.
        y_cur = y_init
        k = 1
        t_sample = 0.
        W = np.zeros(np.shape(self.S)[1]) # create array to store reacton rates
        while t_cur<tmax and k-1<max_steps and stop_cond(y_cur) != True:


            self.rates(y_cur,W) # obtain reaction rates
            rate_tot = np.sum(W) # get total rate
            if rate_tot<=0.:
                print("Error: non-positive rate_tot in gillespie ")
                break
            tau = np.random.exponential(1./rate_tot) # time until next reaction

            # find which reaction happens
            r = np.random.rand()
            rate_sum = W[0]
            ind = 0
            while r>rate_sum/rate_tot:
                ind += 1
                rate_sum += W[ind]

            # update state
            y_cur += self.S[:,ind]
            t_cur += tau
            t_sample += tau

            # sample system state
            if t_sample>=dt_sample:
                y[k] = y_cur
                t[k] = t_cur
                k += 1
                t_sample = 0.

        return t[0:k],y[0:k]

    def ensemble(self,tmax,L,y_init,*,max_steps=10**10,\
            stop_cond=lambda x: False,dt_sample=0.4):
        """"
        Generate ensemble of gillespie simulations

        Parameters
        ----------
        tmax : float
            maximum time to run simulation
        L : number of simulations to run
        y_init_dist: distribution of initial conditions

        Returns
        -------
        t: array
            the time points at which reactions occured
        Y: list of arrays with the state of the system at reaction events for
            each trajectory
        """

        Y = []
        for k in range(L):
            y0 = y_init()
            t,y = self.gillespie(tmax,y0, \
                dt_sample=dt_sample, \
                max_steps=max_steps, \
                stop_cond=stop_cond)
            Y.append(y)
        lens = [len(y) for y in Y]
        Y = np.array([y[:min(lens)] for y in Y]).T
        t = t[:min(lens)]
        return t,Y
