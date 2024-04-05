# import required bibliograghies

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
import time
from numba import njit


## functions for lattice simulation ##

@njit
def initialize_lattice(shape, q):
    '''initialize a random lattice in the q-Potts-model'''
    return np.random.randint(1, q+1, size=shape)

def show_config(config, q): 
    '''shows the lattice graphically'''
    normalized_config = config/q
    pic = Image.fromarray(np.uint8(cm.gist_rainbow(normalized_config)*255)) #toplogical colours with gist_earth
    pic.show()

@njit
def energy_local(config, site_pos, J):
    '''calculates the energy of a local site'''
    d1, d2 = np.shape(config)[0], np.shape(config)[1]
    i, j = site_pos[0], site_pos[1]
    site_spin = config[i ,j]
    nn = [config[(i + 1) % d1, j], config[i, (j + 1) % d2], config[(i - 1) % d1, j], config[i, (j - 1) % d2]]
    site_interaction = 0 
    for e in nn:
        if site_spin == e:
            site_interaction += 1
    return -J*site_interaction

@njit
def energy(config, J):
    '''calculates the energy of a lattice configuration'''
    d1, d2 = np.shape(config)[0], np.shape(config)[1]
    nn_interaction_term = 0
    for i in range(d1): 
        for j in range(d2):
            site_spin = config[i ,j]
            nn = [config[(i+1) % d1, j], config[i, (j+1) % d2]]
            site_interaction = 0 
            for e in nn:
                if site_spin == e:
                    site_interaction += 1
            nn_interaction_term += site_interaction
    return -J*nn_interaction_term

@njit
def magnetization(config, q):
    '''calculates the total magnetization of a lattice configuration'''
    N = config.size
    spin_vectors = np.exp(2j * np.pi * (config - 1) / q)
    M = 1/N * np.sum(spin_vectors)
    return np.abs(M), np.angle(M)

@njit
def magnetization_components(config, q):
    '''calculates the number of each component of the magnetization of lattice configuration'''
    components = []
    for qi in range(1, q+1):
        components.append(np.sum(config == qi))
    return components


## algorithms for simulation ##

@njit
def metropolis_local(config, J, T, q, kB=1):
    '''local metropolis algorithm for one random site'''
    i, j = np.random.randint(0, config.shape[0]), np.random.randint(0, config.shape[1])
    proposed_config = np.copy(config)
    proposed_config[i, j] = np.random.randint(1, q+1)
    proposed_delta_energy = energy_local(proposed_config, [i, j], J) - energy_local(config, [i, j], J)
    if np.random.rand() < np.exp(-(proposed_delta_energy)/(kB*T)):
        return proposed_config, proposed_delta_energy, 1
    else:
        return config, 0, 0

@njit
def metropolis(shape, q, J, T, sweeps, kB=1):
    '''single metropolis algorithm which uses the local metropolis algorithm'''
    config = initialize_lattice(shape, q)
    E = energy(config, J)
    Es = [E]
    M, angle = magnetization(config, q)
    Ms = [M]
    angles = [angle]
    comps = [magnetization_components(config, q)]
    counter = 1
    counter_for_acceptance_rate = 0
    while counter < sweeps:
        deltaE_total = 0
        for _ in range(shape[0]*shape[1]):
            config, delta_E, c = metropolis_local(config, J, T, q)
            deltaE_total += delta_E
            counter_for_acceptance_rate += c
        # if counter%20 == 0:
        #     normalized_config = config/q
        #     pic = Image.fromarray(np.uint8(cm.gist_rainbow(normalized_config)*255))
        #     pic.save(folder + '%g.PNG' %counter)
        Es.append(Es[-1] + deltaE_total)
        M_i, angle_i = magnetization(config, q)
        Ms.append(M_i)
        angles.append(angle_i)
        comps.append(magnetization_components(config, q))
        counter += 1
        # print(100/hits * counter, " %")
    accep_rate = 100/(sweeps*shape[0]*shape[1]) * counter_for_acceptance_rate
    return config, Es, Ms, angles, comps, accep_rate

def single_metropolis_Ts(shape, q, J, Ts, sweeps, j, N, kB=1):
    '''runs the Metropolis algorithm at different temperatures, collects the data and measures the time'''
    fields = []
    Es_T = []
    Ms_T = []
    angles_T = []
    M_comps_T = []
    accep_rate_T = []
    # t0 = time.time()
    for i, T in enumerate(Ts):
        field, Es, Ms, angles, M_comps, accep_rate = metropolis(shape, q, J, T, sweeps, kB)
        fields.append(field)
        Es_T.append(Es)
        Ms_T.append(Ms)
        angles_T.append(angles)
        M_comps_T.append(M_comps)
        accep_rate_T.append(accep_rate)
        print(round(100/(N*len(Ts)) * (i+1+len(Ts)*j), 1), " %")
    # t1 = time.time()
    # print((t1-t0)/60, " min for the simulation")
    return fields, Es_T, Ms_T, angles_T, M_comps_T, accep_rate_T

def parallel_tempering_with_time(shape, q, J, Ts, sweeps, j, N, kB=1):
    '''runs the parallel tempering algorithm and measures the time'''
    t0 = time.time()
    replicas, Es_T, Ms_T, angles_T, M_components_T, replica_bookkeeping, counter = parallel_tempering(shape, q, J, Ts, sweeps, j, N, kB)
    t1 = time.time()
    print((t1-t0)/60, " min for the simulation of lattice %g" %(j+1))
    print("%f %% of the times was a temperature-exchange!" %(100/(sweeps * len(Ts)) * counter))
    return replicas, Es_T, Ms_T, angles_T, M_components_T, replica_bookkeeping

@njit
def parallel_tempering(shape, q, J, Ts, sweeps, j, N, kB=1):
    '''parallel tempering algorithm which uses the local metropolis algorithm'''
    replicas = [initialize_lattice(shape, q) for _ in range(len(Ts))]
    # replicas = [] 
    # same_start = initialize_lattice(shape,q)
    # for i in range(len(Ts)):
    #     replicas.append(same_start)
    counter = 0
    replica_bookkeeping = []
    Es_T = []
    Ms_T = []
    angles_T = []
    M_components_T = []
    counter_for_acceptance_rate_T = []
    accep_rate_T = []
    for i, T in enumerate(Ts):
        replica_bookkeeping.append([T])
        Es_T.append([energy(replicas[i], J)])
        M_T, angle_T = magnetization(replicas[i], q)
        Ms_T.append([M_T])
        angles_T.append([angle_T])
        M_components_T.append([magnetization_components(replicas[i], q)])
        counter_for_acceptance_rate_T.append(0)
    for sweep in range(sweeps):
        for i, T in enumerate(Ts):
            deltaE_total = 0
            for _ in range(int(shape[0]*shape[1])):
                replicas[i], delta_E, c = metropolis_local(replicas[i], J, T, q)
                deltaE_total += delta_E
                counter_for_acceptance_rate_T[i] += c
            Es_T[i].append(Es_T[i][-1] + deltaE_total)
            M_T, angle_T = magnetization(replicas[i], q)
            Ms_T[i].append(M_T)
            angles_T[i].append(angle_T)
            M_components_T[i].append(magnetization_components(replicas[i], q))
        i = 0
        while i < len(Ts)-1:
            beta1, beta2 = 1/(kB*Ts[i]), 1/(Ts[i+1]*kB)
            exponent = (beta1 - beta2) * (-Es_T[i][-1] + Es_T[i + 1][-1])
            if np.random.rand() < np.exp(-exponent):
                counter += 1
                replica_bookkeeping[i].append(replica_bookkeeping[i + 1][-1])
                replica_bookkeeping[i+1].append(replica_bookkeeping[i][-2])
                replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
                Es_T[i].append(Es_T[i+1][-1])
                Es_T[i+1].append(Es_T[i][-2])
                Ms_T[i].append(Ms_T[i+1][-1])
                Ms_T[i+1].append(Ms_T[i][-2])
                angles_T[i].append(angles_T[i+1][-1])
                angles_T[i+1].append(angles_T[i][-2])
                M_components_T[i].append(M_components_T[i+1][-1])
                M_components_T[i+1].append(M_components_T[i+1][-2])
                i += 2
                if i == len(Ts)-1:
                    replica_bookkeeping[-1].append(replica_bookkeeping[-1][-1])
            else:
                replica_bookkeeping[i].append(replica_bookkeeping[i][-1])
                if i == len(Ts)-2:
                    replica_bookkeeping[-1].append(replica_bookkeeping[-1][-1])
                i += 1
        print(round(100/(N*sweeps) * (sweep+j*sweeps), 1), " %")
    for counter_for_acceptance_rate in counter_for_acceptance_rate_T:
        accep_rate_T.append(100/(sweeps*shape[0]*shape[1]) * counter_for_acceptance_rate)
    return replicas, Es_T, Ms_T, angles_T, M_components_T, replica_bookkeeping, counter, accep_rate_T


## statistics ##

@njit
def jackknife_avrg(dataset, n_blocks=50):
    '''jackknife method to calculate averages'''
    raw_avrg = np.mean(dataset)
    sum_1 = 0
    block_size = len(dataset) // n_blocks
    jackknife_samples = np.empty(n_blocks)
    for i in range(n_blocks):
        block_start = i * block_size
        block_end = (i + 1) * block_size
        block_data = np.delete(dataset, slice(block_start, block_end))
        jackknife_samples[i] = np.mean(block_data)
        sum_1 += (jackknife_samples[i] - raw_avrg)**2
    sigma = np.sqrt((n_blocks - 1)/n_blocks * sum_1)
    avrg_bias = 1/n_blocks * np.sum(jackknife_samples)
    true_avrg = raw_avrg - (n_blocks - 1)*(avrg_bias - raw_avrg)
    return true_avrg, sigma

@njit
def jackknife_var(dataset, n_blocks=50):
    '''jackknife method to calculate variances'''
    raw_var = np.var(dataset)
    sum_1 = 0
    block_size = len(dataset) // n_blocks
    jackknife_samples = np.empty(n_blocks)
    for i in range(n_blocks):
        block_start = i * block_size
        block_end = (i + 1) * block_size
        block_data = np.delete(dataset, slice(block_start, block_end))
        jackknife_samples[i] = np.var(block_data)
        sum_1 += (jackknife_samples[i] - raw_var)**2
    sigma = np.sqrt((n_blocks - 1)/n_blocks * sum_1)
    var_bias = 1/n_blocks * np.sum(jackknife_samples)
    true_var = raw_var - (n_blocks - 1)*(var_bias - raw_var)
    return true_var, sigma

@njit
def jackknife_autocorr(dataset, t, n_blocks=10):
    '''jackknife method to calculate autocorrelations'''
    raw_autocorr = autocorrelation_t(dataset, t)
    sum_1 = 0
    block_size = len(dataset) // n_blocks
    jackknife_samples = np.empty(n_blocks)
    for i in range(n_blocks):
        block_start = i * block_size
        block_end = (i + 1) * block_size
        block_data = np.delete(dataset, slice(block_start, block_end))
        jackknife_samples[i] = autocorrelation_t(block_data, t)
        sum_1 += (jackknife_samples[i] - raw_autocorr)**2
    sigma = np.sqrt((n_blocks - 1)/n_blocks * sum_1)
    bias = 1/n_blocks * np.sum(jackknife_samples)
    true_autocorr = raw_autocorr - (n_blocks - 1)*(bias - raw_autocorr)
    return true_autocorr, sigma

@njit
def autocorrelation_t(dataset, t):
    '''calculates the autocorrelation of two datapoints which are separated by t data-elements'''
    stop = len(dataset)-t
    numerator = 0
    for i in range(0, stop):
        numerator += dataset[i]*dataset[i+t]
    numerator *= 1/stop
    return (numerator - np.mean(dataset)**2)/np.var(dataset)

def autocorrelation_ts(datasets, ts):
    '''uses the autocorrelation_t function to calculate the autocorrelation for different t with jackknive routine'''
    #t0 = time.time()
    Cs = []
    Cs_error = []
    for i, dataset in enumerate(datasets):
        Cs.append([])
        Cs_error.append([])
        for t in ts:
            Cs_i, Cs_error_i = jackknife_autocorr(np.array(dataset), t)
            Cs[i].append(Cs_i)
            Cs_error[i].append(Cs_error_i)
    #t1 = time.time()
    #print((t1-t0)/60, " min for the calculation of autocorrelations!")
    return Cs, Cs_error

@njit
def autocorrelation_tau(dataset):
    '''calculates the the autocorrelation tau'''
    tau = 0.5
    for t in range(1, len(dataset)):
        tau += autocorrelation_t(dataset, t)
    return tau


## calculation of observables ##

@njit
def heat_capacity(energies, T, kB=1):
    '''uses the jackknive_var function to calculate the heat capacity'''
    var, error = jackknife_var(energies)
    return 1/(kB * T**2) * var, 1/(kB * T**2) * error

@njit
def magnetic_susceptibility(magnetizations, T, kB=1):
    '''uses the jackknive_var function to calculate the magnetic susceptibility'''
    var, error = jackknife_var(magnetizations)
    return 1/(kB * T) * var, 1/(kB * T) * error

def calc_cVs_Xis(q, J, Ts, Es_T, Ms_T, angles_T):
    '''uses the heat_capacity and the magnetic_susceptibility function to calculate the susceptibilities for different temperatures'''
    cVs = []
    cVs_error = []
    Xis = []
    Xis_error = []
    #t0 = time.time()
    for E_T, M_T, angle_T, T in zip(Es_T, Ms_T, angles_T, Ts):
        cV, cV_error = heat_capacity(np.array(E_T), T)
        Xi, Xi_error = magnetic_susceptibility(np.array(M_T), T)
        cVs.append(cV)
        cVs_error.append(cV_error)
        Xis.append(Xi)
        Xis_error.append(Xi_error)
    #t1 = time.time()
    #print((t1-t0)/60, " min for the calculation of cV and Xi with errors!")
    return cVs, cVs_error, Xis, Xis_error

def calc_Os(data_Ts, string):
    '''uses the jackknive_avrg function to calculate some observables at different temperatures'''
    O_Ts = []
    O_Ts_error = []
    #t0 = time.time()
    for data_T in data_Ts:
        O_T, O_T_error = jackknife_avrg(np.array(data_T))
        O_Ts.append(O_T)
        O_Ts_error.append(O_T_error)
    #t1 = time.time()
    #print((t1-t0)/60, " min for the calculation of " + string + " with errors!")
    return O_Ts, O_Ts_error

def calc_components(q, data_Ts):
    '''uses the jackknive_avrg function to calculate the components of the magnetization'''
    M_comp_Ts = []
    M_comp_Ts_error = []
    #t0 = time.time()
    for k, data in enumerate(data_Ts):
        M_comp_Ts.append([])
        M_comp_Ts_error.append([])
        data_sorted = np.sort(data, axis=1)
        for qi in range(q):
            avrg_component, error_component = jackknife_avrg(data_sorted[:, qi])
            M_comp_Ts[k].append(avrg_component)
            M_comp_Ts_error[k].append(error_component)
    #t1 = time.time()
    #print((t1-t0)/60, " min for the calculation of magnetic components with errors!")
    return M_comp_Ts, M_comp_Ts_error


## make plots ##

def plot_E_M_a(Es, Ms, angles, shape, method, q, J, T, folder, savefigs, openfigs):
    '''plots the Energy, Magnetization and it's direction with respect to simulation time'''
    fig,axs = plt.subplots(3, 1, constrained_layout=True, dpi=150)
    axs[0].plot(Es, '.',markersize=0.2)
    axs[0].set_xlabel('iterations')
    axs[0].set_title('Energy-Trend')
    axs[0].set_ylabel('energy')
    axs[1].plot(Ms, '.',markersize=0.2)
    axs[1].set_xlabel('iterations')
    axs[1].set_title('Magnetic-Trend')
    axs[1].set_ylabel('magnetization')
    axs[2].plot(angles, '.',markersize=0.2)
    axs[2].set_xlabel('iterations')
    axs[2].set_title('Magnetic-Direction-Trend')
    axs[2].set_ylabel('direction')
    fig.suptitle("%s \n lattice size = %g x %g, q = %g, J = %g, T = %f" %(method, shape[0], shape[1], q, J, T))
    plt.show()
    if savefigs:
        plt.savefig(folder + '\\%s_Trends_simulation_%g x %g_q=%g_J=%g_T=%f.png' %(method, shape[0], shape[1], q, J, T), dpi=200, bbox_inches='tight')
    if not openfigs:
        plt.close()

def plot_cV_Xi(shapes, method, q, J, Ts, CVS, CVS_Error, XIS, XIS_Error, folder, savefigs, openfigs):
    '''plots the susceptibilities with respect to temperature'''
    fig,axs = plt.subplots(2, 1, constrained_layout=True, dpi=150)
    fig.suptitle("%s \n susceptibilities, q = %g, J = %g" %(method, q, J))
    for cVs, cVs_error, shape in zip(CVS, CVS_Error, shapes):
        axs[0].errorbar(Ts, cVs, yerr=cVs_error, fmt='.--', linewidth=0.7, capsize=5, capthick=0.7, label='lattice size = %g x %g' %(shape[0], shape[1]))
    axs[0].set_xlabel('temperature')
    axs[0].set_ylabel('heat capacity')
    axs[0].legend()
    for Xis, Xis_error, shape in zip(XIS, XIS_Error, shapes):
        axs[1].errorbar(Ts, Xis, yerr=Xis_error, fmt='.--', linewidth=0.7, capsize=5, capthick=0.7, label='lattice size = %g x %g' %(shape[0], shape[1]))
    axs[1].set_xlabel('temperature')
    axs[1].set_ylabel('magnetic susceptibility')
    axs[1].legend()
    plt.show()
    if savefigs:
        plt.savefig(folder + '\\%s_susceptibilities_over_temperature_q=%g_J=%g.png' %(method, q, J), dpi=200, bbox_inches='tight')
    if not openfigs:
        plt.close()

def plot_O_over_T(shapes, method, q, J, Ts, Os, Os_error, string, folder, savefigs, openfigs):
    '''plots some observable with respect to temperature'''
    fig = plt.figure(dpi=250)
    for Osi, Osi_error, shape in zip(Os, Os_error, shapes):
        plt.errorbar(Ts, Osi, yerr=Osi_error, fmt='.--', linewidth=0.7, capsize=5, capthick=0.7, label='lattice size = %g x %g' %(shape[0], shape[1]))
    plt.xlabel('temperature')
    plt.ylabel(string)
    plt.title("%s \n %s q = %g, J = %g" %(method, string, q, J))
    plt.legend()
    plt.show()
    if savefigs:
        plt.savefig(folder + '\\%s_%s_over_temperature_q=%g_J=%g.png' %(method, string, q, J), dpi=250, bbox_inches='tight')
    if not openfigs:
        plt.close()

def plot_parallel_tempering_bookkeeping(replicas_Ts, shape, q, J, percentage, folder, savefigs, openfigs):
    '''plots the temperature exchange of the replicas with respect to simulation time'''
    colors = {}
    X = []
    Y = []
    colors_xy = []
    color_positions = {}
    for T in replicas_Ts[:, 0]:
        colors[T] = ('#%06X' % np.random.randint(0, 0xFFFFFF))    
    fig = plt.figure(dpi=150)
    for step in range(len(replicas_Ts[0])):
        for n in range(len(replicas_Ts)):
            for T in colors:
                if replicas_Ts[n, step] == T:
                    color = colors[T]
                    if step == 0:
                        X.append(step)
                        Y.append(n)
                        colors_xy.append(color)
                        plt.hlines(n, step - 1/4, step + 1/4, colors=color, label='T = %f' % T)
                        if T not in color_positions:
                            color_positions[T] = []
                        color_positions[T].append((step, n))
                    else:
                        X.append(step)
                        Y.append(n)
                        colors_xy.append(color)
                        plt.hlines(n, step - 1/4, step + 1/4, colors=color)
                        color_positions[T].append((step, n))
    for T, positions in color_positions.items():
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            plt.plot([x1 + 1/4, x2 - 1/4], [y1, y2], linestyle='--', color=colors[T], linewidth=0.7)
    plt.xlabel('simulation time')
    plt.ylabel('replica')
    plt.title('Parallel Tempering - Evolution of Temperatures of the replicas \n %f %% of the times was a temperature-exchange \n in the simulation of lattice with size %g x %g, q=%g and J=%g!' %(percentage, shape[0], shape[1], q, J))
    plt.legend(loc='right')
    plt.show()
    if savefigs:
        plt.savefig(folder + '\\paralle_bookkeeping_%g x %g_q=%g_J=%g.png' %(shape[0], shape[1], q, J), dpi=200, bbox_inches='tight')
    if not openfigs:
        plt.close()

def plot_histogram_Os(Os, q, J, T, string, shape, method, folder, savefigs, openfigs, bins=200):
    '''makes a histogram of some observable'''
    fig = plt.figure(dpi=150)
    plt.hist(Os, bins=bins)
    plt.title('%s, Histogram \n lattice size = %g x %g, q = %g, J = %g, T = %f' %(method, shape[0], shape[1], q, J, T))
    plt.xlabel(string)
    plt.show()
    if savefigs:
        plt.savefig(folder + '\\%s_histogram_%s_%g x %g_q=%g_J=%g_T=%f.png' %(method, string, shape[0], shape[1], q, J, T), dpi=200, bbox_inches='tight')
    if not openfigs:
        plt.close()

def plot_magnetic_components(M_comp_Ts, M_comp_Ts_error, Ts, q, J, shape, method, folder, savefigs, openfigs):
    '''plots the number of each component of the magnetization with respect to temperature'''
    fig = plt.figure(dpi=150)
    for i in range(len(M_comp_Ts[0])):
        plt.errorbar(Ts, np.array(M_comp_Ts)[:, i], yerr=np.array(M_comp_Ts_error)[:, i], fmt='.--', linewidth=0.7, capsize=5, capthick=0.7)
    plt.title('%s \n components of magnetization, lattice size = %g x %g, q = %g, J = %g' %(method, shape[0], shape[1], q, J))
    plt.xlabel('temperature')
    plt.ylabel('# of components of the magnetization')
    plt.show()
    if savefigs:
        plt.savefig(folder + '\\%s_magnetic_comps_%g x %g_q=%g_J=%g.png' %(method, shape[0], shape[1], q, J), dpi=200, bbox_inches='tight')
    if not openfigs:
        plt.close()

def plot_autocorrelation(Ts, ts, Cs, Cs_error, q, shape, method, folder, savefigs, openfigs):
    '''plots the autocorrelation of the dataset for different spacings ts'''
    fig = plt.figure(dpi=150)
    for T, Cs_T, Cs_error_T in zip(Ts, Cs, Cs_error):
        plt.errorbar(ts, Cs_T, yerr=Cs_error_T, fmt='.--', linewidth=0.7, capsize=5, capthick=0.7, label='T = %g' %T)
    plt.title('%s \n Autocorrelation of Energies, lattice size = %g x %g, q = %g' %(method, shape[0], shape[1], q))
    plt.xlabel('monte carlo step')
    plt.ylabel('autocorrelation')
    plt.legend()
    plt.show()
    if savefigs:
        plt.savefig(folder + '\\%s_autocorrelation_%g x %g_q=%g.png' %(method, shape[0], shape[1], q), dpi=200, bbox_inches='tight')
    if not openfigs:
        plt.close()

def make_plots(shapes, method, q, J, Ts, ES_T, MS_T, Angles_T,
               ES_T_data, MS_T_data, Angles_T_data,
               M_Comps=None, M_Comps_Error=None, mag_com=False,
               MS=None, MS_Error=None, mag=False,
               US=None, US_Error=None, inner_energy=False,
               CVS=None, CVS_Error=None, XIS=None, XIS_Error=None, cV_Xi=False,
               Temp_Book=None, percentages=None, parallel_bookkeeping=False,
               ts=None, CS=None, CS_Error=None, autocorr=False,
               t_h_list=None, trends=False, hist=False,
               folder=None, savefigs=False, openfigs=True):
    '''compact function to make plots'''
    t0 = time.time()
    if mag_com:
        for Ms_comps, Ms_comps_error, shape in zip(M_Comps, M_Comps_Error, shapes):
            plot_magnetic_components(Ms_comps, Ms_comps_error, Ts, q, J, shape, method, folder, savefigs, openfigs)
    if mag:
        plot_O_over_T(shapes, method, q, J, Ts, MS, MS_Error, "magnetization", folder, savefigs, openfigs)
    if inner_energy:
        plot_O_over_T(shapes, method, q, J, Ts, US, US_Error, "inner energy", folder, savefigs, openfigs)
    if cV_Xi:
        plot_cV_Xi(shapes, method, q, J, Ts, CVS, CVS_Error, XIS, XIS_Error, folder, savefigs, openfigs)
    if parallel_bookkeeping:
        for temp_book, shape, percentage in zip(Temp_Book, shapes, percentages):
            plot_parallel_tempering_bookkeeping(np.array(temp_book)[:, :50], shape, q, J, percentage, folder, savefigs, openfigs)
    if autocorr:
        for Cs, Cs_error, shape in zip(CS, CS_Error, shapes):
            plot_autocorrelation(Ts, ts, Cs, Cs_error, q, shape, method, folder, savefigs, openfigs)
    if hist:
        for Es_T_data, Ms_T_data, shape in zip(ES_T_data, MS_T_data, shapes):
            for Es, T in zip(Es_T_data[t_h_list[0]:t_h_list[1]], Ts[t_h_list[0]:t_h_list[1]]):
                plot_histogram_Os(Es, q, J, T, 'energy', shape, method, folder, savefigs, openfigs)
            for Ms, T in zip(Ms_T_data[t_h_list[0]:t_h_list[1]], Ts[t_h_list[0]:t_h_list[1]]):
                plot_histogram_Os(Ms, q, J, T, 'magnetization', shape, method, folder, savefigs, openfigs)
    if trends:
        for Es_T, Ms_T, angles_T, shape in zip(ES_T_data, MS_T_data, Angles_T_data, shapes):
            for E_T, M_T, angle_T, T in zip(Es_T[t_h_list[0]:t_h_list[1]], Ms_T[t_h_list[0]:t_h_list[1]],
                                            angles_T[t_h_list[0]:t_h_list[1]], Ts[t_h_list[0]:t_h_list[1]]):
                plot_E_M_a(E_T, M_T, angle_T, shape, method, q, J, T, folder, savefigs, openfigs)
    t1 = time.time()
    t = (t1-t0)/60
    print(t, " min for all plots")
    return t


## run whole program ##

def run_complete_simulation(shapes, q, J, Ts, sweeps, data_start, ts, parallel_or_single = "Parallel Tempering Algorithm", 
                            openfigs=True, savefigs=True, savedata=True, writefile=True,
                            folder = 'C:\\Users\\unter\\Documents\\Uni\\Master Physics\\1. Semester\\Monte Carlo Methods VU\\Project\\first_test_simulation'):
    '''runs a complete simulation and makes all plots'''
    # initializing the lists
    ES_T = []
    ES_T_data = []
    MS_T = []
    MS_T_data = []
    Angles_T = []
    Angles_T_data = []
    M_Components_T = []
    M_Components_T_data = []
    Accep_Rate_T = []
    Replica_Bookkeeping = []
    percentages = []
    US = []
    US_Error = []
    MS = []
    MS_Error = []
    M_Comps = []
    M_Comps_Error = []
    CVS = []
    CVS_Error = []
    XIS = []
    XIS_Error = []
    CS = []
    CS_Error = []
    
    # start the simulation with parallel tempering or single metropolis algorithm
    t0 = time.time()
    for i, shape in enumerate(shapes):
        if parallel_or_single == 'Parallel Tempering Algorithm':
            fields, Es_T, Ms_T, angles_T, M_components_T, replica_bookkeeping, counter, accep_rate_T = parallel_tempering(shape, q, J, Ts, sweeps, i, len(shapes))
            Replica_Bookkeeping.append(replica_bookkeeping)
            percentages.append(100/(sweeps * len(Ts)) * counter)
            parallel_bookkeeping = True
        elif parallel_or_single == 'Single Metropolis Algorithm':
            fields, Es_T, Ms_T, angles_T, M_components_T, accep_rate_T = single_metropolis_Ts(shape, q, J, Ts, sweeps, i, len(shapes))
            parallel_bookkeeping = False
        ES_T.append(Es_T)
        MS_T.append(Ms_T)
        Angles_T.append(angles_T)
        M_Components_T.append(M_components_T)
        Accep_Rate_T.append(accep_rate_T)
        
        # just using the data after thermalization
        ES_T_data.append([np.array(Es[data_start:])*1/(shape[0]*shape[1]) for Es in Es_T])
        MS_T_data.append([Ms[data_start:] for Ms in Ms_T])
        Angles_T_data.append([angles[data_start:] for angles in angles_T])
        M_Components_T_data.append([M_comps[data_start:] for M_comps in M_components_T])
    t1 = time.time()
    print('\n')
    print((t1-t0)/60, " min for all simulations")
    
    
    # print statemant for acceptance rate
    print('\nacceptance rate |   temperature  |   lattice size    ')
    for i, shape in enumerate(shapes):
        for j, T in enumerate(Ts):
            print('%.2f %%         |   %f     |   %g x %g         ' %(Accep_Rate_T[i][j], T, shape[0], shape[1]))

    # print statemant for parallel tempering
    if parallel_or_single == 'Parallel Tempering Algorithm':
        print('\ntemperature exchange |     lattice size')
        for percentage, shape in zip(percentages, shapes):
            print('%f %%          |      %g x %g' %(percentage, shape[0], shape[1]))
    print('\n')
    
    
    # calculation of observables
    t2 = time.time()
    for i, shape in enumerate(shapes):
        Us, Us_error = calc_Os(ES_T_data[i], "inner energy")
        US.append(Us)
        US_Error.append(Us_error)
        Ms, Ms_error = calc_Os(MS_T_data[i], "magnetization")
        MS.append(Ms)
        MS_Error.append(Ms_error)
        M_comps, M_comps_error = calc_components(q, M_Components_T_data[i])
        M_Comps.append(M_comps)
        M_Comps_Error.append(M_comps_error)
        cVs, cVs_error, Xis, Xis_error = calc_cVs_Xis(q, J, Ts, ES_T_data[i], MS_T_data[i], Angles_T_data[i])
        CVS.append(cVs)
        CVS_Error.append(cVs_error)
        XIS.append(Xis)
        XIS_Error.append(Xis_error)
        Cs, Cs_error = autocorrelation_ts(ES_T_data[i], ts)
        CS.append(Cs)
        CS_Error.append(Cs_error)
    t3 = time.time()
    print((t3-t2)/60, " min for all calculations")

    # make and save the plots
    t_h_list = [0, -1]
    time_for_plots = make_plots(shapes, parallel_or_single, q, J, Ts, ES_T, MS_T, Angles_T,
                ES_T_data, MS_T_data, Angles_T_data,
                M_Comps=M_Comps, M_Comps_Error=M_Comps_Error, mag_com=True,
                MS=MS, MS_Error=MS_Error, mag=True,
                US=US, US_Error=US_Error, inner_energy=True,
                CVS=CVS, CVS_Error=CVS_Error, XIS=XIS, XIS_Error=XIS_Error, cV_Xi=True,
                Temp_Book=Replica_Bookkeeping, percentages=percentages, parallel_bookkeeping=parallel_bookkeeping,
                ts=ts, CS=CS, CS_Error=CS_Error, autocorr=True,
                t_h_list=t_h_list, trends=True, hist=True,
                folder=folder, savefigs=savefigs, openfigs=openfigs)
    
    # create text document
    if writefile:
        with open(folder + '\\%s_infos_of_simulation.txt' %method, 'w') as file:
            file.write(method + '\nq = %g, J = %g\n' %(q, J))
            file.write('lattice sizes: ')
            for shape in shapes:
                file.write('%g x %g, ' %(shape[0], shape[1]))
            file.write('\nTemperatures: ')
            for T in Ts:
                file.write('%f, ' %T)
            file.write('\nsweeps = %g \ndata_start = %g \n' %(sweeps, data_start))
            file.write('\n%f min for all simulations \n' %((t1-t0)/60))
            file.write('\nacceptance rate |   temperature  |   lattice size\n')
            for i, shape in enumerate(shapes):
                for j, T in enumerate(Ts):
                    file.write('%.2f %%         |   %f     |   %g x %g\n' %(Accep_Rate_T[i][j], T, shape[0], shape[1]))
            if parallel_or_single == 'Parallel Tempering Algorithm':
                file.write('\n')
                file.write('temperature exchange |     lattice size\n')
                for percentage, shape in zip(percentages, shapes):
                    file.write('%f %%          |      %g x %g\n' %(percentage, shape[0], shape[1]))
            file.write('\n%f min for all calculations \n' %((t3-t2)/60))
            file.write('%f min for all plots' % time_for_plots)
    
    # save data
    if savedata:
        np.save(folder + '\\shapes.npy', np.array(shapes, dtype=object))
        np.save(folder + '\\Ts.npy', np.array(ES_T_data, dtype=object))
        np.save(folder + '\\ES_T_data.npy', np.array(ES_T_data, dtype=object))
        np.save(folder + '\\MS_T_data.npy', np.array(MS_T_data, dtype=object))
        np.save(folder + '\\Angles_T_data.npy', np.array(Angles_T_data, dtype=object))
        np.save(folder + '\\M_Components_T_data.npy', np.array(M_Components_T_data, dtype=object))
        if parallel_or_single == 'Parallel Tempering Algorithm':
            np.save(folder + '\\Replica_Bookkeeping.npy', np.array(Replica_Bookkeeping, dtype=object))
            np.save(folder + '\\percentages.npy', np.array(percentages, dtype=object))


if __name__=='__main__':

    # setting for simulation
    shapes = [(10, 10), (20, 20)]
    q = 5
    J = 1
    Ts = np.array([0.83, 0.84, 0.845, 0.848, 0.85, 0.852, 0.854, 0.856, 0.858, 0.86, 0.862, 0.865, 0.87, 0.88])
    sweeps = 100000
    data_start = 5000
    ts = [4, 8, 16, 32, 64, 128, 256, 512]
    
    method = 'Parallel Tempering Algorithm'
    # method = 'Single Metropolis Algorithm'
    
    folder = 'C:\\Users\\unter\\Documents\\Uni\\Master Physics\\1. Semester\\Monte Carlo Methods VU\\Project\\second_test_simulation'
    
    run_complete_simulation(shapes, q, J, Ts, sweeps, data_start, ts,
                            parallel_or_single=method, openfigs=False, savefigs=True,
                            savedata=False, writefile=True, folder=folder)
    
    
    
    