from glob import glob
import numpy as np
import matplotlib.pyplot as plt

import wobble
from tqdm import tqdm

import h5py
import tensorflow as tf
tf.__version__

date = '200907'
file_num = 1 # initial test

K_t = 2

reg_star_file = '../regularization/EXPRES/55cnc_expres_star_K0.hdf5'
reg_t_file    = f'../regularization/EXPRES/55cnc_expres_t_K{K_t}.hdf5'

# Learning Rates
lr_star_file  = '../data/LearningRates/55cnc_expres_star_K0.hdf5'
lr_t_file     = f'../data/LearningRates/55cnc_expres_t_K{K_t}.hdf5'
lnrs = h5py.File(lr_star_file, 'r')
lnrt = h5py.File(lr_t_file, 'r')


star_name = '55cnc_expres'
print(f'FINDING RVS FOR  {star_name}')
data = wobble.Data(f'../data/{star_name}.hdf5', orders=np.arange(45,70))
data.trim_bad_edges()
#data = wobble.Data(f'../data/{star_name}.hdf5', orders=np.arange(20,86))
results = wobble.Results(data=data)
for r in range(len(data.orders)):
    print('starting order {0} of {1}'.format(r+1, len(data.orders)))
    model = wobble.Model(data, results, r)
    model.add_star('star', regularization_par_file=reg_star_file,
                   learning_rate_template=lnrs['learning_rate_template'][r], rv_steps=20)
    #model.add_telluric('tellurics', variable_bases=2)
    model.add_telluric('tellurics', regularization_par_file=reg_t_file, variable_bases=K_t,
                       learning_rate_template=lnrt['learning_rate_template'][r])
    wobble.optimize_order(model,save_history=True,min_dnll=1e-4,niter=1500,return_best_iter=True)
#results.combine_orders('star')
#results.write_rvs('star', f'./Results/{date}_{star_name}_rvs{file_num}.txt')
results.write(f'./Results/{date}_{star_name}_results{file_num}.hdf5')