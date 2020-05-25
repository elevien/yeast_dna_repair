import pandas as pd
import numpy as np
import os
import data_processing as dp
import single_cell_inference as sc



data_directory = os.getcwd() + '/../../forEthan/'
data = dp.get_data(data_directory)

for experiment in data[1:]:
    print('fitting '+experiment['experiment'])
    sc.constant_break_rate_inference(experiment)
