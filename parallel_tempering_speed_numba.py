import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from matplotlib import cm
import time
from numba import njit

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
    counter = 1
    while counter < sweeps:
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
        # print(100/sweeps * counter, " %")
    return config, Es, Ms, angles


def heat_capacity(energies, T, kB=1):
    var, error = jackknife_var(energies)
    return 1/(kB * T**2) * var, 1/(kB * T**2) * error

def magnetic_susceptibility(magnetizations, T, kB=1):
    var, error = jackknife_var(magnetizations)
    return 1/(kB * T) * var, 1/(kB * T) * error


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

def phase_transitions(shape, q, J, Ts, sweeps, data_start=800000, kB=1):
    cVs = []
    cVs_error = []
    Xis = []
    Xis_error = []
    for T in Ts:
        field, Es, Ms, angles = metropolis(shape, q, J, T, sweeps, kB)
        plot_E_M_a(Es, Ms, angles, q, J, T)
        cV, cV_error = heat_capacity(Es[data_start:], T, kB)
        cVs.append(cV)
        cVs_error.append(cV_error)
        Xi, Xi_error = magnetic_susceptibility(Ms[data_start:], T, kB)
        Xis.append(Xi)
        Xis_error.append(Xi_error)
        print(100/len(Ts) * (Ts.index(T)+1), " %")
    fig,axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].errorbar(Ts, cVs, yerr=cVs_error, fmt='.--', capsize=5, capthick=1)
    axs[0].set_xlabel('T')
    axs[0].set_ylabel('c_V')
    axs[1].errorbar(Ts, Xis, yerr=Xis_error, fmt='.--', capsize=5, capthick=1)
    axs[1].set_xlabel('T')
    axs[1].set_ylabel('Xi')
    plt.title("q = %g, J = %g" %(q, J))
    plt.show()

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

@njit
def parallel_tempering(shape, q, J, Ts, sweeps, kB=1):
    replicas = [initialize_lattice(shape, q) for _ in range(len(Ts))]
    # replicas = [] 
    # same_start = initialize_lattice(shape,q)
    # for i in range(len(Ts)):
    #     replicas.append(same_start)
    counter = 0
    Es_T = []
    Ms_T = []
    angles_T = []
    for i, T in enumerate(Ts):
        Es_T.append([energy(replicas[i], J)])
        M_T, angle_T = magnetization(replicas[i], q)
        Ms_T.append([M_T])
        angles_T.append([angle_T])
    for step in range(sweeps):
        print(100/sweeps * step, " %")
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
                    Es_T[-1].append(Es_T[-1][-1])
                    Ms_T[-1].append(Ms_T[-1][-1])
                    angles_T[-1].append(angles_T[-1][-1])
            else:
                Es_T[i].append(Es_T[i][-1])
                Ms_T[i].append(Ms_T[i][-1])
                angles_T[i].append(angles_T[i][-1])
                i += 1
    print(counter)
    return replicas, Es_T, Ms_T, angles_T

def plots_parallel_tempering(q, J, Ts, Es_T, Ms_T, angles_T, data_start):
    cVs = []
    cVs_error = []
    Xis = []
    Xis_error = []
    for E_T, M_T, angle_T, T in zip(Es_T, Ms_T, angles_T, Ts):
        cV, cV_error = heat_capacity(E_T[data_start:], T)
        Xi, Xi_error = magnetic_susceptibility(M_T[data_start:], T)
        plot_E_M_a(E_T, M_T, angle_T, q, J, T)
        cVs.append(cV)
        cVs_error.append(cV_error)
        Xis.append(Xi)
        Xis_error.append(Xi_error)
    fig,axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].errorbar(Ts, cVs, yerr=cVs_error, fmt='.--', capsize=5, capthick=1)
    axs[0].set_xlabel('T')
    axs[0].set_ylabel('c_V')
    axs[1].errorbar(Ts, Xis, yerr=Xis_error, fmt='.--', capsize=5, capthick=1)
    axs[1].set_xlabel('T')
    axs[1].set_ylabel('Xi')
    plt.title("q = %g, J = %g" %(q, J))
    plt.show()

shape = (20, 20)
q = 5
J = 1
T = 0.8
Ts = np.array([0.6 + i*0.05 for i in range(10)])
sweeps = 1000000
data_start = 800000


# parallel tempering # #
t0 = time.time()
fields, Es_T, Ms_T, angles_T = parallel_tempering(shape, q, J, Ts, sweeps)
t1 = time.time()
print((t1-t0)/60)

plots_parallel_tempering(q, J, Ts, Es_T, Ms_T, angles_T, data_start)



# # metropolis for phase transition # #
# t0 = time.time()
# phase_transitions(shape, q, J, Ts, sweeps, data_start)
# t1 = time.time()
# print((t1-t0)/60)

# print('without number: ', 0.9463132143020629)
# print('with number: ', )
# print(0.9463132143020629/0.22277793486913045, 'x-times faster')


# # metropolis # #
# t0 = time.time()
# field, Es, Ms, angles = metropolis(shape, q, J, T, sweeps)
# t1 = time.time()
# print((t1-t0)/60, "min")

# show_config(field, q)
# plot_E_M_a(Es, Ms, angles, q, J, T)

# cV = heat_capacity(Es[data_start:], T)
# Xi = magnetic_susceptibility(Ms[data_start:], T)

# print(cV)
# print(Xi)

# fig = plt.figure(dpi=150)
# plt.hist(Es[800000:], bins=100)
# fig = plt.figure(dpi=150)
# plt.hist(Ms[800000:], bins=100)


