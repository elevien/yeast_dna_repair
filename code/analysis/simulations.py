import copy,math
import numpy as np
import pandas as pd



class bellman_harris_model_base:
    def __init__(self,f,f0,Q,type_names):
        '''
        This is a base class for implementing multi-type population growth models.
        can be sub-classes (see below) to implement specific models, such as
        constant probability model and constant rate model, but also more complex
        models with intermediate states and sub-populations with different
        growth rates. Note that in order to make the code general purpose
        and user friendly, I have sacraficed some computational efficiency.


        Paramaters
        -----------
        f - f is the generation time distribution, which takes 3 arguments:
                * the mother cell generation time
                * the current time
                * the population size
                * the cell type
        f0 - the lag time (i.e. the distribution of the first cell's
            generation time). This takes no arguments

        Q - a function which takes two arguments
                * the mother cell generation time
                * current time as arguments and
            it returns a num_types dimensional square matrix
            with rows that sum to 1



        '''
        self.f = f
        self.f0 = f0
        self.Q = Q # the state transition probability , a matrix function of t
        self.num_types = len(type_names) # get number of types
        self.type_names = type_names


    def run_well(self,Nmax,*,dt_sample=0.01,tmax=100):
        '''
        Simulate a single well

        Paramaters
        -----------
            Nmax          - the maximum number of cells to generate


        Opational Paramaters
        -----------
            tmax        - the maximum time to run
            dt_sample   - frequancy to save samples of the number of cells


        Output
        -----------
            output - a pandas dataframe containing the numbers of cells of each type
                    in its columns

        '''

        gt0 = self.f0()
        cells_gt = np.zeros(Nmax) # generation times
        cells_dt = np.zeros(Nmax) # division times
        cells_type = np.zeros(Nmax,dtype=int) # type of cell (0 if no cell in slot)
        cells_gt[0] = gt0
        cells_dt[0] = gt0
        cells_type[0] = 0

        N = [1]
        M = [np.zeros(self.num_types,dtype=int)] # M is the array of the number of each type
        M[0][0] = 1  # by convention, the initial cell is always type 0
        n=1
        T = [0.]
        t_sample = 0.
        n = 1  # current number of cells
        m = np.zeros(self.num_types,dtype=int) # current number of each type
        m[0] = 1
        t = 0. # the currect time (i.e. the time at which the last event occured)


        while n<Nmax-1 and t<tmax:
            # get the cell which is dividing
            ind = np.argmin(cells_dt[0:n])
            mother_dt = cells_dt[ind]       # time when this cell divides
            mother_gt = cells_gt[ind]       # generation time of cell
            mother_type =  cells_type[ind]  # type of cells
            t = mother_dt                   # update the currect time


            # update our saved arrays
            t_last = T[-1]
            while t-t_last>dt_sample:
                t_last += dt_sample
                T.append(t_last)
                N.append(n)
                M.append([])
                for k in range(self.num_types):
                   M[-1].append(m[k])


            # use the transition matrix Q to select the type of of the daughter cells
            daughter_type = np.random.choice(range(self.num_types),p=self.Q(mother_gt,t)[mother_type])
            cells_type[n] = daughter_type
            cells_type[ind] = daughter_type
            if daughter_type==mother_type:
                # if daughter has the same type as the mother we gain 1 cell
                # of that type
                m[daughter_type] = m[daughter_type]+1
            else:
                # other wise the number of cells of the mother type descreses
                # by one and we gain 2 cells of the daughter type
                m[daughter_type] = m[daughter_type]+2
                m[mother_type] = m[mother_type]-1

            n = n + 1 # total number of cells always increases by one

            # draw the generation times of the daughter cells
            gt1 = self.f(mother_gt,t,n,daughter_type)
            gt2 = self.f(mother_gt,t,n,daughter_type)

            # and save their generation times and division times
            cells_gt[ind] = gt1
            cells_dt[ind] = t + gt1
            cells_gt[n] = gt2
            cells_dt[n] = t + gt2

        # put the output in a dataframe
        output = pd.DataFrame({"time":T,"bf":N})


        M = np.array(M)
        for k in range(self.num_types): # add the number of each type
            output[self.type_names[k]] = M[:,k]

        return output

    def run_ensemble(self,Nwells,Nmax,**kwargs):
        '''
        Simulate ensemble of wells

        Paramaters
        -----------
            Nwells        - the number of wells to simulate
            Nmax          - the maximum number of cells to generate in each well


        Opational Paramaters
        -----------
            kwargs        - all the keyword arguments for run_well



        Output
        -----------
            output - a pandas dataframe containing the numbers of cells of each type
                    in its columns along with the number of the well associated with
                    each sample

        '''
        outputs = []
        for k in range(Nwells):
            output = self.run_well(Nmax,**kwargs)
            output["well"] = np.ones(len(output),dtype=int)*k
            outputs.append(output)

        # truncate so they all have the same length, like the data
        min_length = np.min([len(df) for df in outputs])
        outputs = [df[0:min_length] for df in outputs]
        data = pd.concat(outputs)
        return data


# ----------------------------------------------------------------------------
# pre-defined models
# ----------------------------------------------------------------------------

class constant_probability_model(bellman_harris_model_base):
    def __init__(self,f,f0,p):
        self.p = p
        type_names = ['ng','gfp']
        Q = lambda gt,t: np.array([[1-p,p],[0,1]])
        bellman_harris_model_base.__init__(self,f,f0,Q,type_names)

class constant_rate_model(bellman_harris_model_base):
    def __init__(self,f,f0,beta):
        self.beta = beta
        type_names = ['ng','gfp']

        Q = lambda gt,t: np.array([[np.exp(-beta*gt),1-np.exp(-beta*gt)],[0,1]])
        bellman_harris_model_base.__init__(self,f,f0,Q,type_names)

# class two_state_constant_probability_model(bellman_harris_model_base):
#     def __init__(self,f,f0,beta):
#         self.beta = beta
#         type_names = ['ng','broken','gfp']
#
#         p = lambda gt:1-np.exp(-beta*gt)
#         Q = lambda gt,t: np.array([[np.exp(-beta*gt),1-np.exp(-beta*gt)],[0,1]])
#         bellman_harris_model_base.__init__(self,f,f0,Q,type_names)
