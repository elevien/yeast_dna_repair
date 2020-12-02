import pickle
import pystan
import os
import pymc3 # why do I need this to compile stan models??
import sys



fname = str(sys.argv[1])
model_name = fname.split('.')[0]
model  = pystan.StanModel(file='./stan_models/'+fname)
with open('./stan_models/'+model_name+'.pkl', 'wb') as f:
    pickle.dump(model, f)
