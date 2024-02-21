#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 21:31:32 2023

@author: maximilianpfandner
"""

import numpy as np 
from numba import njit
import random as ran
import matplotlib.pyplot as plt

# @njit
def spin_grid_generator(dimension,elevation = 1): 
    grid = np.random.choice([-1,1],size=dimension)
    return grid 

# spin_grid_generator((100,100))

def show_field(field,elevation): 
    from PIL import Image
    pic = Image.fromarray(np.uint8((field + elevation)) * 0.5 * 255)
    pic.show()

@njit
def energy(field,d1,d2,J = 1,H = 0):
    ''' calculates the energy of a state'''
    nn_interaction_term = 0
    magnetic_term = 0 
    for i in range(d1): 
        for j in range(d2): # creates sum over all spins 
            i_pb = i #% d1 # ensures periodic bounderies 
            j_pb = j #% d2 # ensures periodic bounderies 
            single_spin = field[i_pb ,j_pb]
            single_interaction = 0
            nn = [field[(i_pb + 1) % d1,j_pb],field[i_pb,(j_pb + 1) % d2]] # additional sum over next nearest neighbours 
            for a in nn: 
                single_interaction += a*field[i,j] # i, j sum 
            magnetic_term += single_spin # i sum 
            nn_interaction_term += single_interaction
    spin_netto_interaction = J * nn_interaction_term
    spin_netto_magnetic = H * magnetic_term
    return spin_netto_interaction + spin_netto_magnetic  

@njit 
def metropolis(field,J = 1,H = 0,T = 1e-15,kb = 1.380649e-23,count = 100000): # achtung niedrige temperatur und verhältnismäßig starkes feld 
    d1,d2 = np.shape(field)
    counter = 0 
    while counter <= count: 
        choiced1 = ran.randint(0,d1-1)
        choiced2 = ran.randint(0,d2-1)
        start_spin = field[choiced1,choiced2]
        newfield = field.copy()
        newfield[choiced1,choiced2] = - start_spin 
        delta_E = energy(field,d1,d2,J,H) - energy(newfield,d1,d2,J,H)
        r = ran.uniform(0, 1)
        if delta_E > 0: 
             if r < np.exp(-delta_E/(T*kb)): 
                field = newfield 
        else: 
            field = newfield
        counter += 1
    return field

# test_field = spin_grid_generator((50,50))
# show_field(test_field, 1)
# show_field(metropolis(test_field), 1)

# now for 3d 

# def energy_3d(field,d1,d2,d3,J,H):
#     ''' calculates the energy of a state'''
#     nn_interaction_term = 0
#     magnetic_term = 0 
#     for i in range(d1-1): 
#         for j in range(d2-1): # creates sum over all spins 
#             for k in range(d3-1): 
#                 i_pb = i % d1 # ensures periodic bounderies 
#                 j_pb = j % d2 # ensures periodic bounderies 
#                 k_pb = k % d3 # ensures periodic bounderies 
#                 single_spin = field[i_pb ,j_pb, k_pb]
#                 single_interaction = 0
#                 nn = [field[i_pb + 1,j_pb,k_pb],field[i_pb,j_pb + 1,k_pb],field[i_pb,j_pb,k_pb + 1]] # additional sum over next nearest neighbours 
#                 for a in nn: 
#                     single_interaction += a*single_spin # i, j, k sum 
#                 magnetic_term += single_spin # i sum 
#                 nn_interaction_term += single_interaction
#     spin_netto_interaction = J * nn_interaction_term
#     spin_netto_magnetic = H * magnetic_term
#     return spin_netto_interaction + spin_netto_magnetic 

# def metropolis(field,J = 1,H = 0,T = 1,kb = 1.380649e-23,count = 10000): # achtung niedrige temperatur und verhältnismäßig starkes feld 
#     d1,d2,d3 = np.shape(field)
#     counter = 0 
#     while counter <= count: 
#         choiced1 = ran.randint(0,d1-1)
#         choiced2 = ran.randint(0,d2-1)
#         choiced3 = ran.randint(0,d3-1)
#         start_spin = field[choiced1,choiced2,choiced3]
#         newfield = field.copy()
#         newfield[choiced1,choiced2,choiced3] = - start_spin 
#         delta_E = energy_3d(field,d1,d2,J,H) - energy_3d(newfield,d1,d2,d3,J,H)
#         r = ran.uniform(0, 1)
#         if delta_E > 0: 
#              if r < np.exp(-delta_E/(T*kb)): 
#                 field = newfield 
#         else: 
#             field = newfield
#         counter += 1
#     return field
    
# # make it more efficient 

# def energy_fast_total(field,d1,d2,J = 1,H = 0):
#     import scipy.ndimage as sn
#     import scipy.signal as ss
#     kernel = sn.generate_binary_structure(2,1) #no diagonal elements are neigbours 
#     kernel[1][1] = False
#     nn_part = -field * sn.convolve(field,kernel, mode = 'constant',cval=0.0)
#     nn_interaction = J * nn_part.sum()
#     magnetic_term = 0 
#     for i in range(d1-1): 
#         for j in range(d2-1): 
#             magnetic_term += field[i,j]
#     sumn = np.sum(nn_part)     
#     magnetic_interaction = H * magnetic_term
#     return nn_interaction  + magnetic_interaction 
    
def magnetization(field):
    return np.sum(field)
    
@njit
def energy_local(field,d1,d2,i,j,J,H):
    ''' calculates the energy of a state'''
    nn_interaction_term = 0
    magnetic_term = 0 
    i_pb = i #% (d1 - 1) # ensures periodic bounderies 
    j_pb = j #% (d2 - 1) # ensures periodic bounderies 
    single_spin = field[i_pb ,j_pb]
    nn = [field[(i_pb + 1) % d1,j_pb],field[i_pb,(j_pb + 1) % d2],field[(i_pb - 1) % d1,j_pb],field[i_pb,(j_pb - 1) % d2]] # additional sum over next nearest neighbours 
    for a in nn: 
        nn_interaction_term += a*field[i,j] 
    magnetic_term = single_spin 
    spin_netto_interaction = J * nn_interaction_term
    spin_netto_magnetic = H * magnetic_term
    return spin_netto_interaction + spin_netto_magnetic 
# @njit
def metropolis_local(field,J = 1,H = 0,T = 1,kb = 1,count = 10000000): # achtung niedrige temperatur und verhältnismäßig starkes feld 
    d1,d2 = np.shape(field)
    counter = 0
    initial_energy = energy(field, d1, d2, J, H)
    initial_magnetization = magnetization(field)
    energy_trend = [initial_energy]
    magnetic_trend = [initial_magnetization]
    while counter <= count: 
        choiced1 = ran.randint(0,d1-1)
        choiced2 = ran.randint(0,d2-1)
        start_spin = field[choiced1,choiced2]
        newfield = field.copy()
        newfield[choiced1,choiced2] = - start_spin 
        delta_E = energy_local(field,d1,d2,choiced1,choiced2,J,H) - energy_local(newfield,d1,d2,choiced1,choiced2,J,H)
        r = ran.uniform(0, 1)
        if delta_E > 0: 
             if r < np.exp(-delta_E/(T*kb)): 
                field = newfield 
                initial_energy += -delta_E
                energy_trend.append(initial_energy)
                magnetic_trend.append(magnetization(field))
        else: 
            field = newfield
            initial_energy += -delta_E
            energy_trend.append(initial_energy)
            magnetic_trend.append(magnetization(field))
        counter += 1
    return field, energy_trend, magnetic_trend

# test_field = spin_grid_generator((5,5))     
# d1,d2 = np.shape(test_field)   
# print(energy(test_field,d1,d2))
# print(energy_fast_total(test_field,d1,d2))

# test_field = spin_grid_generator((30, 30))
# print(np.shape(test_field))
# show_field(test_field, 1)
# print(magnetization(test_field))


# field, energy_trend, magnetic_trend = metropolis_local(test_field)

# show_field(field, 1)

# fig = plt.figure(dpi=150)
# plt.plot(energy_trend, '.',markersize=0.2)
# plt.xlabel('iterations')
# plt.title('Energy-Trend')

# fig = plt.figure(dpi=150)
# plt.plot(magnetic_trend, '.',markersize=0.2)
# plt.xlabel('iterations')
# plt.title('Magnetic-Trend')

# print(np.var(energy_trend[10000:]))
# print(np.var(magnetic_trend[10000:]))


J_values = [0.7 + i*0.02 for i in range(20)]
print(J_values)

cv = []
xi = []

for J in J_values:
    init_field = spin_grid_generator((30, 30))
    field, energy_trend, magnetic_trend = metropolis_local(init_field, J)
    cv.append(np.var(energy_trend[30000:]))
    xi.append(np.var(magnetic_trend[30000:]))
    print("lol")


fig = plt.figure(dpi=150)
plt.plot(J_values, cv)

fig = plt.figure(dpi=150)
plt.plot(J_values, xi)






