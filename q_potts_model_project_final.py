
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
import time
from numba import njit


## functions for lattice simulation ##

@njit
def initialize_lattice(shape, q):
    return np.random.randint(1, q+1, size=shape)

def show_config(config, q): 
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
    '''calculates the energy of a configuration'''
    d1, d2 = np.shape(config)[0], np.shape(config)[1]
    nn_interaction_term = 0
    for i in range(d1): 
        for j in range(d2): # creates sum over all spins 
            site_spin = config[i ,j]
            nn = [config[(i+1) % d1, j], config[i, (j+1) % d2]] # additional sum over next nearest neighbours 
            site_interaction = 0 
            for e in nn:
                if site_spin == e:
                    site_interaction += 1
            #site_interaction = np.sum(np.equal(site_spin, nn))
            nn_interaction_term += site_interaction
    return -J*nn_interaction_term

@njit
def magnetization(config, q):
    '''calculates the magnetization of a configuration'''
    N = config.size
    spin_vectors = np.exp(2j * np.pi * (config - 1) / q)
    M = 1/N * np.sum(spin_vectors)
    return np.abs(M), np.angle(M)

@njit
def magnetization_components(config, q):
    components = []
    for qi in range(1, q+1):
        components.append(np.sum(config == qi))
    return components


## algorithms for simulation ##

@njit
def metropolis_local(config, J, T, q, kB=1):
    i, j = np.random.randint(0, config.shape[0]), np.random.randint(0, config.shape[1])
    proposed_config = np.copy(config)
    proposed_config[i, j] = np.random.randint(1, q+1)
    proposed_delta_energy = energy_local(proposed_config, [i, j], J) - energy_local(config, [i, j], J)
    if np.random.rand() < np.exp(-(proposed_delta_energy)/(kB*T)):
        return proposed_config, proposed_delta_energy
    else:
        return config, 0

@njit
def metropolis(shape, q, J, T, sweeps, kB=1):
    config = initialize_lattice(shape, q)
    E = energy(config, J)
    Es = [E]
    M, angle = magnetization(config, q)
    Ms = [M]
    angles = [angle]
    comps = [magnetization_components(config, q)]
    counter = 1
    while counter < sweeps:
        deltaE_total = 0
        for _ in range(shape[0]*shape[1]):
            config, delta_E = metropolis_local(config, J, T, q)
            deltaE_total += delta_E
        # if counter%2000 == 0:
        #     title = "C:\\Users\\unter\\Documents\\GitHub\\Master\\pictures for gif\\%g.PNG" %counter
        #     normalized_config = config/q
        #     pic = Image.fromarray(np.uint8(cm.gist_rainbow(normalized_config)*255))
        #     pic.save(title)
        Es.append(Es[-1] + deltaE_total)
        M_i, angle_i = magnetization(config, q)
        Ms.append(M_i)
        angles.append(angle_i)
        comps.append(magnetization_components(config, q))
        counter += 1
        # print(100/hits * counter, " %")
    return config, Es, Ms, angles, comps

def single_metropolis_Ts(shape, q, J, Ts, sweeps, kB=1):
    fields = []
    Es_T = []
    Ms_T = []
    angles_T = []
    M_comps_T = []
    t0 = time.time()
    for i, T in enumerate(Ts):
        field, Es, Ms, angles, M_comps = metropolis(shape, q, J, T, sweeps, kB)
        fields.append(field)
        Es_T.append(Es)
        Ms_T.append(Ms)
        angles_T.append(angles)
        M_comps_T.append(M_comps)
        print(round(100/len(Ts) * (i+1), 1), " %")
    t1 = time.time()
    print((t1-t0)/60, " min for the simulation")
    return fields, Es_T, Ms_T, angles_T, M_comps_T

def parallel_tempering_with_time(shape, q, J, Ts, sweeps, kB=1):
    t0 = time.time()
    replicas, Es_T, Ms_T, angles_T, M_components_T, replica_bookkeeping, counter = parallel_tempering(shape, q, J, Ts, sweeps, kB)
    t1 = time.time()
    print((t1-t0)/60, " min for the simulation")
    print("%f %% of the times was a temperature-exchange!" %(100/(sweeps * len(Ts)) * counter))
    return replicas, Es_T, Ms_T, angles_T, M_components_T, replica_bookkeeping

@njit
def parallel_tempering(shape, q, J, Ts, sweeps, kB=1):
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
    for i, T in enumerate(Ts):
        replica_bookkeeping.append([T])
        Es_T.append([energy(replicas[i], J)])
        M_T, angle_T = magnetization(replicas[i], q)
        Ms_T.append([M_T])
        angles_T.append([angle_T])
        M_components_T.append([magnetization_components(replicas[i], q)])
    for sweep in range(sweeps):
        for i, T in enumerate(Ts):
            deltaE_total = 0
            for _ in range(int(shape[0]*shape[1])):
                replicas[i], delta_E = metropolis_local(replicas[i], J, T, q)
                deltaE_total += delta_E
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
        print(round(100/sweeps * sweep, 1), " %")
    return replicas, Es_T, Ms_T, angles_T, M_components_T, replica_bookkeeping, counter


## statistics ##

@njit
def jackknife_avrg(dataset, n_blocks=50):
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
def autocorrelation_t(dataset, t):
    stop = len(dataset)-t
    numerator = 0
    for i in range(0, stop):
        numerator += dataset[i]*dataset[i+t]
    numerator *= 1/stop
    return (numerator - np.mean(dataset)**2)/np.var(dataset)

def autocorrelation_ts(datasets, ts):
    t0 = time.time()
    Cs = []
    for i, dataset in enumerate(datasets):
        Cs.append([])
        for t in ts:
            Cs[i].append(autocorrelation_t(np.array(dataset), t))
    t1 = time.time()
    print((t1-t0)/60, " min for the calculation of autocorrelations!")
    return Cs

@njit
def autocorrelation_tau(dataset):
    tau = 0.5
    for t in range(1, len(dataset)):
        tau += autocorrelation_t(dataset, t)
    return tau


## calculation of observables ##

@njit
def heat_capacity(energies, T, kB=1):
    var, error = jackknife_var(energies)
    return 1/(kB * T**2) * var, 1/(kB * T**2) * error

@njit
def magnetic_susceptibility(magnetizations, T, kB=1):
    var, error = jackknife_var(magnetizations)
    return 1/(kB * T) * var, 1/(kB * T) * error

def calc_cVs_Xis(q, J, Ts, Es_T, Ms_T, angles_T):
    cVs = []
    cVs_error = []
    Xis = []
    Xis_error = []
    t0 = time.time()
    for E_T, M_T, angle_T, T in zip(Es_T, Ms_T, angles_T, Ts):
        cV, cV_error = heat_capacity(np.array(E_T), T)
        Xi, Xi_error = magnetic_susceptibility(np.array(M_T), T)
        cVs.append(cV)
        cVs_error.append(cV_error)
        Xis.append(Xi)
        Xis_error.append(Xi_error)
    t1 = time.time()
    print((t1-t0)/60, " min for the calculation of cV and Xi with errors!")
    return cVs, cVs_error, Xis, Xis_error

def calc_Os(data_Ts, string):
    O_Ts = []
    O_Ts_error = []
    t0 = time.time()
    for data_T in data_Ts:
        O_T, O_T_error = jackknife_avrg(np.array(data_T))
        O_Ts.append(O_T)
        O_Ts_error.append(O_T_error)
    t1 = time.time()
    print((t1-t0)/60, " min for the calculation of " + string + " with errors!")
    return O_Ts, O_Ts_error

def calc_components(q, data_Ts):
    M_comp_Ts = []
    M_comp_Ts_error = []
    t0 = time.time()
    for k, data in enumerate(data_Ts):
        M_comp_Ts.append([])
        M_comp_Ts_error.append([])
        data_sorted = np.sort(data, axis=1)
        for qi in range(q):
            avrg_component, error_component = jackknife_avrg(data_sorted[:, qi])
            M_comp_Ts[k].append(avrg_component)
            M_comp_Ts_error[k].append(error_component)
    t1 = time.time()
    print((t1-t0)/60, " min for the calculation of magnetic components with errors!")
    return M_comp_Ts, M_comp_Ts_error


## make plots ##

def plot_E_M_a(Es, Ms, angles, q, J, T):
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
    fig.suptitle("q = %g, J = %g, T = %g" %(q, J, T))

def plot_cV_Xi(q, J, Ts, cVs, cVs_error, Xis, Xis_error):
    fig,axs = plt.subplots(2, 1, constrained_layout=True, dpi=150)
    fig.suptitle("susceptibilities, q = %g, J = %g" %(q, J))
    axs[0].errorbar(Ts, cVs, yerr=cVs_error, fmt='.--', capsize=5, capthick=1)
    axs[0].set_xlabel('temperature')
    axs[0].set_ylabel('c_V')
    axs[1].errorbar(Ts, Xis, yerr=Xis_error, fmt='.--', capsize=5, capthick=1)
    axs[1].set_xlabel('temperature')
    axs[1].set_ylabel('Xi')
    plt.show()

def plot_O_over_T(q, J, Ts, Os, Os_error, string):
    fig = plt.figure(dpi=150)
    plt.errorbar(Ts, Os, yerr=Os_error, fmt='.--', capsize=5, capthick=1)
    plt.xlabel('temperature')
    plt.ylabel(string)
    plt.title(string + " q = %g, J = %g" %(q, J))
    plt.show()

def plot_parallel_tempering_bookkeeping(replicas_Ts):
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
    plt.title('Parallel Tempering - Evolution of Temperatures of the replicas')
    plt.legend(loc='right')
    plt.show()

def plot_histogram_Os(Os, q, J, T, string, bins=200):
    fig = plt.figure(dpi=150)
    plt.hist(Os, bins=bins)
    plt.title('Histogram, q = %g, J = %g, T = %f' %(q, J, T))
    plt.xlabel(string)
    plt.show()

def plot_magnetic_components(M_comp_Ts, M_comp_Ts_error, Ts, q, J):
    fig = plt.figure(dpi=150)
    for i in range(len(M_comp_Ts[0])):
        plt.errorbar(Ts, np.array(M_comp_Ts)[:, i], yerr=np.array(M_comp_Ts_error)[:, i], fmt='.--', capsize=5, capthick=1)
    plt.title('components of magnetization, q = %g, J = %g' %(q, J))
    plt.xlabel('temperature')
    plt.ylabel('# of the same component')
    plt.show()

def plot_autocorrelation(Ts, ts, Cs):
    fig = plt.figure(dpi=150)
    for T, Cs_T in zip(Ts, Cs):
        plt.plot(ts, Cs_T, '.--', label='T = %g' %T)
    plt.title('Autocorrelation of Energies')
    plt.xlabel('monte carlo step')
    plt.ylabel('autocorrelation')
    plt.legend()
    plt.show()

def make_plots(q, J, Ts, fields, Es_T, Ms_T, angles_T,
               Es_T_data, Ms_T_data, angles_T_data,
               M_comps=None, M_comps_error=None, mag_com=False,
               Ms=None, Ms_error=None, mag=False,
               Us=None, Us_error=None, inner_energy=False,
               cVs=None, cVs_error=None, Xis=None, Xis_error=None, cV_Xi=False,
               temp_book=None, parallel_bookkeeping=False,
               ts=None, Cs=None, autocorr=False,
               t_h_list=[0, -1], trends=False, hist=False):
    t0 = time.time()
    if mag_com:
        plot_magnetic_components(M_comps, M_comps_error, Ts, q, J)
    if mag:
        plot_O_over_T(q, J, Ts, Ms, Ms_error, "magnetization")
    if inner_energy:
        plot_O_over_T(q, J, Ts, Us, Us_error, "inner energy")
    if cV_Xi:
        plot_cV_Xi(q, J, Ts, cVs, cVs_error, Xis, Xis_error)
    if parallel_bookkeeping:
        plot_parallel_tempering_bookkeeping(np.array(temp_book)[:, :50])
    if autocorr:
        plot_autocorrelation(Ts, ts, Cs)
    if hist:
        for Es, T in zip(Es_T_data[t_h_list[0]:t_h_list[1]], Ts[t_h_list[0]:t_h_list[1]]):
            plot_histogram_Os(Es, q, J, T, 'energy')
        for Ms, T in zip(Ms_T_data[t_h_list[0]:t_h_list[1]], Ts[t_h_list[0]:t_h_list[1]]):
            plot_histogram_Os(Ms, q, J, T, 'magnetization')
    if trends:
        for E_T, M_T, angle_T, T in zip(Es_T[t_h_list[0]:t_h_list[1]], Ms_T[t_h_list[0]:t_h_list[1]],
                                        angles_T[t_h_list[0]:t_h_list[1]], Ts[t_h_list[0]:t_h_list[1]]):
            plot_E_M_a(E_T, M_T, angle_T, q, J, T)
    t1 = time.time()
    print((t1-t0)/60, " min for all plots")


## setting for simulation ##

shape = (20, 20)
q = 5
J = 1
Ts = np.array([0.66 + i*0.04 for i in range(11)])
sweeps = 100000
data_start = 5000
ts = [4, 8, 16, 32, 64, 128, 256, 512]


## single-metropolis or parallel tempering ##

# parallel tempering # 
fields, Es_T, Ms_T, angles_T, M_components_T, replica_bookkeeping = parallel_tempering_with_time(shape, q, J, Ts, sweeps)
parallel_bookkeeping = True

# metropolis for phase transition #
# fields, Es_T, Ms_T, angles_T, M_components_T = single_metropolis_Ts(shape, q, J, Ts, sweeps)
# replica_bookkeeping = None
# parallel_bookkeeping = False


# just using the data after thermalization #
Es_T_data = [Es[data_start:] for Es in Es_T]
Ms_T_data = [Ms[data_start:] for Ms in Ms_T]
angles_T_data = [angles[data_start:] for angles in angles_T]
M_components_T_data = [M_components[data_start:] for M_components in M_components_T]


# calculation of observables #
Us, Us_error = calc_Os(Es_T_data, "inner energy")
Ms, Ms_error = calc_Os(Ms_T_data, "magnetization")
M_comps, M_comps_error = calc_components(q, M_components_T_data)
cVs, cVs_error, Xis, Xis_error = calc_cVs_Xis(q, J, Ts, Es_T_data, Ms_T_data, angles_T_data)
Cs = autocorrelation_ts(Es_T_data, ts)

# make the plots #
t_h_list = [4, 7]
make_plots(q, J, Ts, fields, Es_T, Ms_T, angles_T,
            Es_T_data, Ms_T_data, angles_T_data,
            M_comps=M_comps, M_comps_error=M_comps_error, mag_com=True,
            Ms=Ms, Ms_error=Ms_error, mag=True,
            Us=Us, Us_error=Us_error, inner_energy=True,
            cVs=cVs, cVs_error=cVs_error, Xis=Xis, Xis_error=Xis_error, cV_Xi=True,
            temp_book=replica_bookkeeping, parallel_bookkeeping=parallel_bookkeeping,
            ts=ts, Cs=Cs, autocorr=True,
            t_h_list=t_h_list, trends=True,  hist=True)



# metropolis with one Temperature #
# T = 0.7
# t0 = time.time()
# field, Es, Ms, angles = metropolis(shape, q, J, T, hits)
# t1 = time.time()
# print((t1-t0)/60, "min")

# show_config(field, q)
# plot_E_M_a(Es, Ms, angles, q, J, T)

# cV = heat_capacity(Es[data_start:], T)
# Xi = magnetic_susceptibility(Ms[data_start:], T)

# print(cV)
# print(Xi)

# plot_histogram_energy(Es, q, T)

