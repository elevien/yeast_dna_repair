import pickle
import pystan
import os
import pymc3 # why do I need this to compile stan models??


#
# model  = pystan.StanModel(file='./constant_break_rate.stan')
# with open('constant_break_rate.pkl', 'wb') as f:
#     pickle.dump(model, f)

model  = pystan.StanModel(file='./per_division_break.stan')
with open('per_division_break.pkl', 'wb') as f:
    pickle.dump(model, f)
