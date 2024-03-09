import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
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
    M = np.sum(spin_vectors)
    M /= N
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
def metropolis(shape, q, J, T, hits, kB=1):
    config = initialize_lattice(shape, q)
    E = energy(config, J)
    Es = [E]
    M, angle = magnetization(config, q)
    Ms = [M]
    angles = [angle]
    counter = 1
    while counter < hits:
        # for _ in range(shape[0]*shape[1]):
        config, delta_E = metropolis_local(config, J, T, q)
        # if counter%2000 == 0:
        #     title = "C:\\Users\\unter\\Documents\\GitHub\\Master\\pictures for gif\\%g.PNG" %counter
        #     normalized_config = config/q
        #     pic = Image.fromarray(np.uint8(cm.gist_rainbow(normalized_config)*255))
        #     pic.save(title)
        Es.append(Es[-1] + delta_E)
        M_i, angle_i = magnetization(config, q)
        Ms.append(M_i)
        angles.append(angle_i)
        counter += 1
        # print(100/hits * counter, " %")
    return config, Es, Ms, angles

def single_metropolis_Ts(shape, q, J, Ts, hits, kB=1):
    fields = []
    Es_T = []
    Ms_T = []
    angles_T = []
    t0 = time.time()
    for i, T in enumerate(Ts):
        field, Es, Ms, angles = metropolis(shape, q, J, T, hits, kB)
        fields.append(field)
        Es_T.append(Es)
        Ms_T.append(Ms)
        angles_T.append(angles)
        print(100/len(Ts) * (i+1), " %")
    t1 = time.time()
    print((t1-t0)/60, " min for the simulation")
    return fields, Es_T, Ms_T, angles_T

def parallel_tempering_with_time(shape, q, J, Ts, hits, kB=1):
    t0 = time.time()
    replicas, Es_T, Ms_T, angles_T, temperatures_bookkeeping, counter = parallel_tempering(shape, q, J, Ts, hits, kB)
    t1 = time.time()
    print((t1-t0)/60, " min for the simulation")
    print("%f %% of the times was a temperature-exchange!" %(100/(hits * len(Ts)) * counter))
    return replicas, Es_T, Ms_T, angles_T, temperatures_bookkeeping

@njit
def parallel_tempering(shape, q, J, Ts, hits, kB=1):
    replicas = [initialize_lattice(shape, q) for _ in range(len(Ts))]
    # replicas = [] 
    # same_start = initialize_lattice(shape,q)
    # for i in range(len(Ts)):
    #     replicas.append(same_start)
    counter = 0
    temperatures_bookkeeping = []
    Es_T = []
    Ms_T = []
    angles_T = []
    for i, T in enumerate(Ts):
        temperatures_bookkeeping.append([T, T])
        Es_T.append([energy(replicas[i], J)])
        M_T, angle_T = magnetization(replicas[i], q)
        Ms_T.append([M_T])
        angles_T.append([angle_T])
    for hit in range(hits):
        print(100/hits * hit, " %")
        for i, T in enumerate(Ts):
            replicas[i], delta_E = metropolis_local(replicas[i], J, T, q)
            Es_T[i].append(Es_T[i][-1] + delta_E)
            M_T, angle_T = magnetization(replicas[i], q)
            Ms_T[i].append(M_T)
            angles_T[i].append(angle_T)
        i = 0
        while i < len(Ts)-1:
            beta1, beta2 = 1/(kB*Ts[i]), 1/(Ts[i+1]*kB)
            exponent = (beta1 - beta2) * (-Es_T[i][-1] + Es_T[i + 1][-1])
            if np.random.rand() < np.exp(-exponent):
                temperatures_bookkeeping[i].append(temperatures_bookkeeping[i + 1][-1])
                temperatures_bookkeeping[i+1].append(temperatures_bookkeeping[i][-2])
                counter += 1
                replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
                Es_T[i].append(Es_T[i+1][-1])
                Es_T[i+1].append(Es_T[i][-2])
                Ms_T[i].append(Ms_T[i+1][-1])
                Ms_T[i+1].append(Ms_T[i][-2])
                angles_T[i].append(angles_T[i+1][-1])
                angles_T[i+1].append(angles_T[i][-2])
                i += 2
                if i == len(Ts)-1:
                    temperatures_bookkeeping[-1].append(temperatures_bookkeeping[-1][-1])
                    Es_T[-1].append(Es_T[-1][-1])
                    Ms_T[-1].append(Ms_T[-1][-1])
                    angles_T[-1].append(angles_T[-1][-1])
            else:
                temperatures_bookkeeping[i].append(temperatures_bookkeeping[i][-1])
                Es_T[i].append(Es_T[i][-1])
                Ms_T[i].append(Ms_T[i][-1])
                angles_T[i].append(angles_T[i][-1])
                if i == len(Ts)-2:
                    temperatures_bookkeeping[-1].append(temperatures_bookkeeping[-1][-1])
                    Es_T[-1].append(Es_T[-1][-1])
                    Ms_T[-1].append(Ms_T[-1][-1])
                    angles_T[-1].append(angles_T[-1][-1])
                i += 1
    return replicas, Es_T, Ms_T, angles_T, temperatures_bookkeeping, counter


## calculation of cV and Xi ##

def heat_capacity(energies, T, kB=1):
    var, error = jackknife_var(energies)
    return 1/(kB * T**2) * var, 1/(kB * T**2) * error

def magnetic_susceptibility(magnetizations, T, kB=1):
    var, error = jackknife_var(magnetizations)
    return 1/(kB * T) * var, 1/(kB * T) * error

def calc_cVs_Xis(q, J, Ts, Es_T, Ms_T, angles_T, data_start):
    cVs = []
    cVs_error = []
    Xis = []
    Xis_error = []
    t0 = time.time()
    for E_T, M_T, angle_T, T in zip(Es_T, Ms_T, angles_T, Ts):
        cV, cV_error = heat_capacity(E_T[data_start:], T)
        Xi, Xi_error = magnetic_susceptibility(M_T[data_start:], T)
        cVs.append(cV)
        cVs_error.append(cV_error)
        Xis.append(Xi)
        Xis_error.append(Xi_error)
    t1 = time.time()
    print((t1-t0)/60, " min for the calculation of cV and Xi with errors!")
    return cVs, cVs_error, Xis, Xis_error


## statistics ##

def jackknife_avrg(dataset, n_blocks=10):
    raw_avrg = np.average(dataset)
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

def autocorrelation_t(dataset, t):
    stop = len(dataset)-t
    numerator = 0
    for i in range(0, stop):
        numerator += dataset[i]*dataset[i+t]
    numerator *= 1/stop
    return (numerator - np.average(dataset)**2)/np.var(dataset)

def autocorrelation_time(dataset):
    tau = 0.5
    for t in range(1, len(dataset)):
        tau += autocorrelation_t(dataset, t)
    return tau


## make plots ##

def plot_E_M_a(Es, Ms, angles, q, J, T):
    fig,axs = plt.subplots(3, 1, constrained_layout=True, dpi=150)
    axs[0].plot(Es, '.',markersize=0.2)
    axs[0].set_xlabel('iterations')
    axs[0].set_title('Energy-Trend')
    axs[0].set_ylabel('Energy')
    axs[1].plot(Ms, '.',markersize=0.2)
    axs[1].set_xlabel('iterations')
    axs[1].set_title('Magnetic-Trend')
    axs[1].set_ylabel('Magnetisation')
    axs[2].plot(angles, '.',markersize=0.2)
    axs[2].set_xlabel('iterations')
    axs[2].set_title('Magnetic-Direction-Trend')
    axs[2].set_ylabel('Direction')
    fig.suptitle("q = %g, J = %g, T = %g" %(q, J, T))

def plot_cV_Xi(Ts, cVs, cVs_error, Xis, Xis_error):
    fig,axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].errorbar(Ts, cVs, yerr=cVs_error, fmt='.--', capsize=5, capthick=1)
    axs[0].set_xlabel('T')
    axs[0].set_ylabel('c_V')
    axs[1].errorbar(Ts, Xis, yerr=Xis_error, fmt='.--', capsize=5, capthick=1)
    axs[1].set_xlabel('T')
    axs[1].set_ylabel('Xi')
    plt.title("q = %g, J = %g" %(q, J))
    plt.show()

def plot_parallel_tempering_bookkeeping(Temperatures):
    num_simulations = len(Temperatures)
    num_steps = len(Temperatures[0])
    fig = plt.figure(dpi=150)
    for i in range(num_simulations):
        plt.plot(range(num_steps), Temperatures[i], ".--", label=f'replica {i + 1}')
    plt.xlabel('Simulation Time')
    plt.ylabel('Temperature')
    plt.title('Parallel Tempering - Evolution of Temperatures of the replicas')
    plt.legend()
    plt.show()

def plot_histogram_energy(Es, q, T, bins=100):
    fig = plt.figure(dpi=150)
    plt.hist(Es[data_start:], bins=bins)
    plt.title('histogram, q = %g, T = %f' %(q, T))
    plt.xlabel('energy')
    plt.show()

def plot_magnetic_components(configs, Ts, q):
    Ys = []
    fig = plt.figure(dpi=150)
    for config in configs:
        components = magnetization_components(config, q)
        Ys.append(sorted(components))
    for i in range(len(Ys[0])):
        plt.plot(Ts, np.array(Ys)[:, i], '.--')
    plt.title('components of magnetization, q = %g' %q)
    plt.xlabel('temperature')
    plt.ylabel('# of the same component')
    plt.show()

def make_plots(q, J, Ts, fields, Es_T, Ms_T, angles_T, cVs, cVs_error, Xis, Xis_error, data_start, temp_book=None, parallel_bookkeeping=False, trends=True, cV_Xi=True, hist=True, mag_com=True):
    t0 = time.time()
    if parallel_bookkeeping:
        plot_parallel_tempering_bookkeeping(np.array(temp_book)[:, :100])
    if trends:
        for E_T, M_T, angle_T, T in zip(Es_T, Ms_T, angles_T, Ts):
            plot_E_M_a(E_T, M_T, angle_T, q, J, T)
    if cV_Xi:
        plot_cV_Xi(Ts, cVs, cVs_error, Xis, Xis_error)
    if hist:
        for Es, T in zip(Es_T, Ts):
            plot_histogram_energy(Es, q, T)
    if mag_com:
        plot_magnetic_components(fields, Ts, q)
    t1 = time.time()
    print((t1-t0)/60, " min for all plots")


## setting for simulation ##

shape = (20, 20)
q = 5
J = 1
Ts = np.array([0.6 + i*0.05 for i in range(10)])
hits = 1000000
data_start = 500000


## single-metropolis or parallel tempering ##

# parallel tempering # 
fields, Es_T, Ms_T, angles_T, temperatures_bookkeeping = parallel_tempering_with_time(shape, q, J, Ts, hits)
cVs, cVs_error, Xis, Xis_error = calc_cVs_Xis(q, J, Ts, Es_T, Ms_T, angles_T, data_start)
make_plots(q, J, Ts, fields, Es_T, Ms_T, angles_T, cVs, cVs_error, Xis, Xis_error, data_start, temp_book=temperatures_bookkeeping, parallel_bookkeeping=True)


# metropolis for phase transition #
# fields, Es_T, Ms_T, angles_T = single_metropolis_Ts(shape, q, J, Ts, hits)
# cVs, cVs_error, Xis, Xis_error = calc_cVs_Xis(q, J, Ts, Es_T, Ms_T, angles_T, data_start)
# make_plots(q, J, Ts, fields, Es_T, Ms_T, angles_T, cVs, cVs_error, Xis, Xis_error, data_start)


## metropolis with one Temperature ##

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


