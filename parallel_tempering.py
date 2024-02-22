import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from matplotlib import cm
import time


def initialize_lattice(shape, q):
    return np.random.randint(1, q+1, size=shape)

def show_config(config, q): 
    normalized_config = config/q
    pic = Image.fromarray(np.uint8(cm.gist_rainbow(normalized_config)*255)) #toplogical colours with gist_earth
    pic.show()

def energy_local(config, site_pos, J):
    '''calculates the energy of a local site'''
    d1, d2 = np.shape(config)[0], np.shape(config)[1]
    i, j = site_pos[0], site_pos[1]
    site_spin = config[i ,j]
    nn = [config[(i + 1) % d1, j], config[i, (j + 1) % d2], config[(i - 1) % d1, j], config[i, (j - 1) % d2]]
    site_interaction = np.sum(np.equal(site_spin, nn)) 
    return -J*site_interaction

def energy(config, J):
    '''calculates the energy of a configuration'''
    d1, d2 = np.shape(config)[0], np.shape(config)[1]
    nn_interaction_term = 0
    for i in range(d1): 
        for j in range(d2): # creates sum over all spins 
            site_spin = config[i ,j]
            nn = [config[(i+1) % d1, j], config[i, (j+1) % d2]] # additional sum over next nearest neighbours 
            site_interaction = np.sum(np.equal(site_spin, nn))
            nn_interaction_term += site_interaction
    return -J*nn_interaction_term

def magnetization(config, q):
    '''calculates the magnetization of a configuration'''
    N = config.size
    spin_vectors = np.exp(2j * np.pi * (config - 1) / q)
    M = np.sum(spin_vectors, axis=None)
    M /= N
    return np.abs(M), np.angle(M)

def metropolis_local(config, J, T, kB=1):
    i, j = np.random.randint(0, config.shape[0]), np.random.randint(0, config.shape[1])
    proposed_config = np.copy(config)
    proposed_config[i, j] = np.random.randint(1, q+1)
    proposed_delta_energy = energy_local(proposed_config, [i, j], J) - energy_local(config, [i, j], J)
    if np.random.rand() < np.exp(-(proposed_delta_energy)/(kB*T)):
        return proposed_config, proposed_delta_energy
    else:
        return config, 0

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
        config, delta_E = metropolis_local(config, J, T)
        # if counter%500 == 0:
        #     title = "C:\\Users\\unter\\Documents\\GitHub\\Master\\pictures for gif\\%g.PNG" %counter
        #     normalized_config = config/q
        #     pic = Image.fromarray(np.uint8(cm.gist_rainbow(normalized_config)*255))
        #     pic.save(title)
        Es.append(Es[-1] + delta_E)
        M_i, angle_i = magnetization(config, q)
        Ms.append(M_i)
        angles.append(angle_i)
        counter += 1
    return config, Es, Ms, angles


def heat_capacity(energies, T, kB=1):
    return 1/(kB * T**2) * np.var(energies)

def magnetic_susceptibility(magnetizations, T, kB=1):
    return 1/(kB * T) * np.var(magnetizations)

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

def phase_transitions(shape, q, J, Ts, sweeps):
    cVs = []
    Xis = []
    for T in Ts:
        field, Es, Ms, angles = metropolis(shape, q, J, T, sweeps)
        plot_E_M_a(Es, Ms, angles, q, J, T)
        cVs.append(heat_capacity(Es[800000:], T))
        Xis.append(magnetic_susceptibility(Ms[800000:], T))
        print("ah")
    fig,axs = plt.subplots(2, 1, constrained_layout=True)
    axs[0].plot(Ts, cVs, '.--')
    axs[0].set_xlabel('T')
    axs[0].set_ylabel('c_V')
    axs[1].plot(Ts, Xis, '.--')
    axs[1].set_xlabel('T')
    axs[1].set_ylabel('Xi')
    plt.title("q = %g, J = %g" %(q, J))
    plt.show()


shape = (20, 20)
q = 5
J = 1
T = 0.851
Ts = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05]
sweeps = 1000000

# t0 = time.time()
# phase_transitions(shape, q, J, Ts, sweeps)
# t1 = time.time()
# print((t1-t0)/60)


t0 = time.time()
field, Es, Ms, angles = metropolis(shape, q, J, T, sweeps)
t1 = time.time()
print((t1-t0)/60, "min")


show_config(field, q)
plot_E_M_a(Es, Ms, angles, q, J, T)

cV = heat_capacity(Es[800000:], T)
Xi = magnetic_susceptibility(Ms[800000:], T)

print(cV)
print(Xi)
