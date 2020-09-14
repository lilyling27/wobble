import numpy as np
import h5py
import os
import wobble

import learningRates

if __name__ == "__main__":
    # change these keywords:
    starname = '101501_expres'
    datafile = '../data/{}.hdf5'.format(starname)
    R = 86 # the number of echelle orders total in the data set
    orders = np.arange(20,80) # list of indices for the echelle orders to be tuned
    K_star = 0 # number of variable components for stellar spectrum
    K_t = 2 # number of variable components for telluric spectrum 
    reg_star_file = f'../regularization/55cnc_expres_star_K{K_star}.hdf5'
    reg_t_file    = f'../regularization/55cnc_expres_t_K{K_t}.hdf5'
    lr_star_file  = '../learning_rates/{0}_star_K{1}.hdf5'.format(starname, K_star)
    lr_t_file     = '../learning_rates/{0}_t_K{1}.hdf5'.format(starname, K_t)
    get_fine = False # Whether to try learning rates between orders of magnitude
    plot = True
    verbose = True # warning: this will print a lot of info & progress bars!
    
    # create directory for plots if it doesn't exist:
    if plot:
        plot_dir = f'../data/learning_rates/{starname}/'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    
    # create learning rate parameter files if they don't exist:
    star_filename = lr_star_file
    if not os.path.isfile(star_filename):
        learningRates.generate_learningRate_file(star_filename, R, type='star')
    tellurics_filename = lr_t_file
    if not os.path.isfile(tellurics_filename):                
        learningRates.generate_learningRate_file(tellurics_filename, R, type='telluric')
        
    # load up the data we'll use for training:
    data = wobble.Data(datafile, orders=orders) # to get N_epochs
    data.trim_bad_edges()
        
    # improve each order's regularization:
    results = wobble.Results(data=data)
    for r,o in enumerate(orders):
        if verbose:
            print('---- STARTING ORDER {0} ----'.format(o))
            print("starting values:")
            print("star:")
            with h5py.File(star_filename, 'r') as f:
                for key in list(f.keys()):
                    print("{0}: {1:.0e}".format(key, f[key][o]))
            print("tellurics:")
            with h5py.File(tellurics_filename, 'r') as f:
                for key in list(f.keys()):
                    print("{0}: {1:.0e}".format(key, f[key][o]))
        
        model = learningRates.modelSetup(data, results, r,
                                         reg_star_file, reg_t_file,
                                         K_star=K_star, K_t=K_t)
        lr_t, lr_s = learningRates.improve_learningRates(model, finer_grid=get_fine,
                                                         plot=plot, plot_dir=plot_dir)
        
        with h5py.File(star_filename, 'r+') as f:
            f['learning_rate_template'][o] = np.copy(lr_s)
        with h5py.File(tellurics_filename, 'r+') as f:
            f['learning_rate_template'][o] = np.copy(lr_t)
        
        if verbose:                                 
            print('---- ORDER {0} COMPLETE ({1}/{2}) ----'.format(o,r,len(orders)-1))
            print("best values:")
            print("star:")
            with h5py.File(star_filename, 'r') as f:
                for key in list(f.keys()):
                    print("{0}: {1:.0e}".format(key, f[key][o]))
            print("tellurics:")
            with h5py.File(tellurics_filename, 'r') as f:
                for key in list(f.keys()):
                    print("{0}: {1:.0e}".format(key, f[key][o]))