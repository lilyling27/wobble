import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
T = tf.float64
from tqdm import tqdm
import h5py
import copy

from wobble.data import Data
from wobble.model import Model
from wobble.results import Results
from wobble.history import History
from wobble.utils import get_session

def generate_learningRate_file(filename, R, type='star'):
    """
    Create a learning rate parameter file with default values.
    
    Parameters
    ----------
    filename : str
        Name of file to be made.
    R : int
        Number of echelle orders.
    type : str, optional
        Type of object; sets which default values to use.
        Acceptable values are 'star' (default) or 'telluric'
    """
    if type=='star':
        learning_rate_par = ['learning_rate_template', 'learning_rate_rvs']
        defaults = [0.1, 1]
    elif type=='telluric':
        learning_rate_par = ['learning_rate_template']
        defaults = [0.01]
    else:
        assert False, 'ERROR: type not recognized.'
    
    with h5py.File(filename,'w') as f:
        for par, val in zip(learning_rate_par, defaults):
            f.create_dataset(par, data=np.zeros(R)+val)

def modelSetup(data, results, r, reg_star_file, reg_t_file,
               K_star=0,K_t=0):
    """
    Initialize model object and run setup
    Sneakily insert a placeholder for the different learning rates
        Allows them to be changed later when the optimizers are run
    """
    model = Model(data, results, r)
    model.add_star('star',regularization_par_file=reg_star_file,variable_bases=K_star)
    model.add_telluric('tellurics', regularization_par_file=reg_t_file,variable_bases=K_t)
    
    model.initialize_templates()
    
    model.synth = tf.zeros(np.shape(model.data.xs[model.r]), dtype=T, name='synth')
    for c in model.components:
        c.setup(model.data, model.r)
        model.synth = tf.add(model.synth, c.synth, name='synth_add_{0}'.format(c.name))
    
    model.nll = 0.5*tf.reduce_sum(tf.square(tf.constant(model.data.ys[model.r], dtype=T) 
                                           - model.synth, name='nll_data-model_sq') 
                                * tf.constant(model.data.ivars[model.r], dtype=T), name='nll_reduce_sum')
    
    for c in model.components:
        model.nll = tf.add(model.nll, c.nll, name='nll_add_{0}'.format(c.name))
    
    model.lr_star_t = tf.compat.v1.placeholder(tf.float32, [], name='star_learning_rate_template')
    model.lr_tell_t = tf.compat.v1.placeholder(tf.float32, [], name='tell_learning_rate_template')
    model.lr_star_r = tf.compat.v1.placeholder(tf.float32, [], name='star_learning_rate_rv')
    
    model.updates = []
    for c in model.components:
        if not c.template_fixed:
            if c.name == 'star':
                placeholder = model.lr_star_t
            elif c.name == 'tellurics':
                placeholder = model.lr_tell_t
            else:
                placeholder = c.learning_rate_template
            c.dnll_dtemplate_ys = tf.gradients(model.nll, c.template_ys)
            c.opt_template = tf.compat.v1.train.AdamOptimizer(learning_rate=placeholder).minimize(model.nll,
                        var_list=[c.template_ys], name='opt_minimize_template_{0}'.format(c.name))
            model.updates.append(c.opt_template)
        if not c.rvs_fixed:
            if c.name == 'star':
                placeholder = model.lr_star_r
            else:
                placeholder = c.learning_rate_rvs
            c.dnll_drvs = tf.gradients(model.nll, c.rvs)
            c.opt_rvs = tf.compat.v1.train.AdamOptimizer(learning_rate=c.learning_rate_rvs,
                                               epsilon=1.).minimize(model.nll,
                        var_list=[c.rvs], name='opt_minimize_rvs_{0}'.format(c.name))
            model.updates.append(c.opt_rvs)
        if c.K > 0:
            c.opt_basis_vectors = tf.compat.v1.train.AdamOptimizer(c.learning_rate_basis).minimize(model.nll,
                        var_list=[c.basis_vectors], name='opt_minimize_basis_vectors_{0}'.format(c.name))
            model.updates.append(c.opt_basis_vectors)
            c.opt_basis_weights = tf.compat.v1.train.AdamOptimizer(c.learning_rate_basis).minimize(model.nll,
                        var_list=[c.basis_weights], name='opt_minimize_basis_weights_{0}'.format(c.name))
            model.updates.append(c.opt_basis_weights)
    
    session = get_session()
    session.run(tf.compat.v1.global_variables_initializer())
    
    return model

def test_learningRate_combo(model, niter=100, rv_opt_steps=20,
                            star_template_lr = 0.1, star_rv_lr=1, 
                            telluric_template_lr=0.01):
    """
    Test a combination of learning rates
    Break if after first iteration the nll gets higher
    """
    # Define feed_dict with new learning rates
    feed_dict = {model.lr_star_t: star_template_lr,
                 model.lr_tell_t: telluric_template_lr,
                 model.lr_star_r: star_rv_lr}
    
    # Optimizing Code
    
    # Test if learning rates are too high
    # i.e. allow for oscillation in the nll
    history = History(model, niter+1)
    history.nll_history.fill(np.inf)
    history.save_iter(model, 0)
    init_nll = history.nll_history[0]
    
    session = get_session()
    # Run optimizer
    for i in range(niter):
        for c in model.components:
            if not c.template_fixed:
                session.run(c.opt_template,feed_dict=feed_dict)
            if c.K > 0:
                session.run(c.opt_basis_vectors,feed_dict=feed_dict)
        for c in model.components:
            if not c.rvs_fixed:
                for _ in range(c.rv_opt_steps):
                    session.run(c.opt_rvs,feed_dict=feed_dict)
            if c.K > 0:
                session.run(c.opt_basis_weights,feed_dict=feed_dict)
        history.save_iter(model, i+1)
    
    return history

def improve_learningRates(model, niter=100, rv_opt_steps=20, plot=False, plot_dir='',
                          init_lr_star_template = 0.1, init_lr_star_rv = 1,
                          init_lr_t_template = 0.01, finer_grid=False):
    """
    Try a grid of learning rates.
    We're just not going to bother with star RV learning rate for now.
    """
    session = get_session()
    lrs_range = np.logspace(1,-4,6)
    result_grid = np.zeros((len(lrs_range)*len(lrs_range),4),dtype=float)
    
    if plot: # initialize array for plotting
        nll_grid = []
    
    for i in tqdm(range(len(lrs_range))):
        lr_t = init_lr_t_template*lrs_range[i]
        for j in range(len(lrs_range)):
            lr_st = init_lr_star_template*lrs_range[::-1][j]
            session.run(tf.compat.v1.global_variables_initializer()) # THIS RESETS THE MODEL???
            history = test_learningRate_combo(model, niter=niter,rv_opt_steps=rv_opt_steps,
                                    star_template_lr=lr_st,telluric_template_lr=lr_t)
            result_grid[i*len(lrs_range)+j] = [lr_t,lr_st, history.nll_history[-1],
                                               history.nll_history[1]-history.nll_history[0]]
            if plot: # Save nlls for plotting
                nll_grid.append(np.copy(history.nll_history))
    result_grid = np.array(result_grid)
    best_lr_t, best_lr_s = result_grid[np.argmin(result_grid[:,2])][:2]
    best_nll = np.min(result_grid[:,2][np.isfinite(result_grid[:,2])])
    worst_nll = np.max(result_grid[:,2][np.isfinite(result_grid[:,2])]) # for ploting only
    
    if plot:
        plt.figure(figsize=(6.4*len(lrs_range),4.8*len(lrs_range)))
        plotGrid(model.data.orders[model.r], result_grid, nll_grid, lrs_range, niter=niter)
        plt.savefig(plot_dir+f'LRtest_o{model.data.orders[model.r]}.png')
        plt.close()
    
    # Re-run grid search on finer grid
    if finer_grid:
        lrs_range_fine = np.logspace(-1,1,9)
        result_grid_fine = np.zeros((len(lrs_range_fine)*len(lrs_range_fine),4),dtype=float)
        
        if plot:
            nll_grid_fine = []
        
        for i in tqdm(range(len(lrs_range_fine))):
            lr_t = best_lr_t*lrs_range_fine[i]
            for j in range(len(lrs_range_fine)):
                lr_st = best_lr_s*lrs_range[::-1][j]
                session.run(tf.compat.v1.global_variables_initializer()) # THIS RESETS THE MODEL???
                history = test_learningRate_combo(model, niter=niter,rv_opt_steps=rv_opt_steps,
                                        star_template_lr=lr_st,telluric_template_lr=lr_t)
                result_grid_fine[i*len(lrs_range)+j] = [lr_t,lr_st,
                    np.where(history.nll_history[-1]<1,np.inf,history.nll_history[-1]<1),
                    history.nll_history[1]-history.nll_history[0]]
                
                if plot:
                    nll_grid_fine.append(np.copy(history.nll_history))
        
        result_grid_fine = np.array(result_grid_fine)
        best_lr_t, best_lr_s = result_grid_fine[np.argmin(result_grid_fine[:,2])][:2]
        best_nll = np.min(result_grid_fine[:,2][np.isfinite(result_grid_fine[:,2])])
        worst_nll = np.max(result_grid_fine[:,2])
    
        if plot:
            plt.figure(figsize=(6.4*len(lrs_range_fine),4.8*len(lrs_range_fine)))
            plotGrid(model.data.orders[model.r], result_grid_fine, nll_grid_fine, lrs_range_fine, niter=niter)
            plt.savefig(plot_dir+f'LRtestFine_o{model.data.orders[model.r]}.png')
            plt.close()
    """ This is garbage, don't know why
    if plot:
        plt.figure()
        plt.title('Final NLL Values')
        plt.xlabel('Stellar Template LR')
        plt.ylabel('Telluric Template LR')
        plt.scatter(result_grid[:,1][np.isfinite(result_grid[:,2])],
                    result_grid[:,0][np.isfinite(result_grid[:,2])],
                    c=np.log10(result_grid[:,2][np.isfinite(result_grid[:,2])]), cmap='plasma_r',
                    vmin=np.log10(best_nll),vmax=np.log10(worst_nll))
        if finer_grid:
            plt.scatter(result_grid_fine[:,1][np.isfinite(result_grid_fine[:,2])],
                        result_grid_fine[:,0][np.isfinite(result_grid_fine[:,2])],
                        c=np.log10(result_grid_fine[:,2][np.isfinite(result_grid_fine[:,2])]), cmap='plasma_r',
                        vmin=np.log10(best_nll),vmax=np.log10(worst_nll))
        plt.colorbar(label='log_10(Final NLL) ')
        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.tight_layout()
        plt.savefig(plot_dir+f'LRtestNLLS_o{model.data.orders[model.r]}.png')
    """
    
    return best_lr_t, best_lr_s

def plotGrid(order, result_grid, nll_grid, test_range,
            niter=100):
    nll_grid = np.array(nll_grid)
    counter=-1
    for tt in range(len(test_range)):
        for st in range(len(test_range)):
            plt.subplot(len(test_range),len(test_range),counter+2)
            counter+=1
            plt.title('Order {}; lr_t {:.3}; lr_s {:.3}'.format(order,
                                                          result_grid[counter][0],
                                                          result_grid[counter][1]))
            plt.plot(nll_grid[counter][np.isfinite(nll_grid[counter])],'.-',alpha=0.5)
            plt.plot(0,nll_grid[counter,0],'ro')
            plt.plot(np.argmin(nll_grid[counter]),nll_grid[counter].min(),'go')
            
            plt.axhline(np.nanmin(nll_grid[np.isfinite(nll_grid)]),color='g')
            plt.xlim(0,niter)
    plt.tight_layout()
    