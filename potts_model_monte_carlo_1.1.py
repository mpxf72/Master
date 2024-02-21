#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:30:05 2024

@author: maximilianpfandner
"""

import numpy as np 
from numba import njit
import random as ran
import matplotlib.pyplot as plt
import cmath

def spin_grid_generator(dimension,q): 
    q_list = []
    for i in range(q): 
        q_list.append(i+1)
    grid = np.random.choice(q_list,size=dimension)
    return grid 

def spin_grid_generator_cold(dimension):
    grid = np.ones(dimension)
    return grid 

def show_field(field,q): 
    from PIL import Image
    from matplotlib import cm
    normalized_field = field/q
    pic = Image.fromarray(np.uint8(cm.gist_rainbow(normalized_field)*255)) #toplogical colours with gist_earth
    pic.show()

@njit
def energy(field,d1,d2,J,H):
    ''' calculates the energy and magnetization of a state'''
    nn_interaction_term = 0
    magnetic_term = 0
    for i in range(d1): 
        for j in range(d2): # creates sum over all spins 
            single_spin = field[i ,j]
            single_interaction = 0
            nn = [field[(i+1) % d1, j], field[i, (j+1) % d2]] # additional sum over next nearest neighbours 
            for a in nn:
                if a == single_spin: # only interaction if site has same state like nn 
                    single_interaction += 1 # i, j sum
            nn_interaction_term += single_interaction
            if single_spin == 1:
                magnetic_term += 1
    spin_netto_interaction = J * nn_interaction_term
    spin_netto_magnetic = H * magnetic_term
    spin_netto_energy = spin_netto_interaction + spin_netto_magnetic
    L_sqaure = (d1*d2) # also for rectengular 
    action_desity = spin_netto_energy/L_sqaure
    return spin_netto_energy, action_desity 

@njit
def magnetization(field, d1, d2, q):
    magnetization = 0
    for qi in range(q+1):   # calculate magnetization in potts model
        magnetization += np.exp(2*np.pi*1j*qi/q) * (field == qi).sum()/(d1*d2)
    return abs(magnetization), cmath.phase(magnetization)

@njit
def energy_local(field,d1,d2,i,j,J,H):
    ''' calculates the energy and magnetization of a local site'''
    nn_interaction_term = 0
    magnetic_term = 0 
    single_spin = field[i ,j]
    single_interaction = 0
    nn = [field[(i + 1) % d1, j], field[i, (j + 1) % d2], field[(i - 1) % d1, j], field[i, (j - 1) % d2]] # additional sum over next nearest neighbours 
    for a in nn:
        if a == single_spin: # only interaction if site has same state like nn 
            single_interaction += 1 # i, j sum  
    nn_interaction_term += single_interaction
    if single_spin == 1:
        magnetic_term += 1  
    magnetic_term = single_spin 
    spin_netto_interaction = J * nn_interaction_term
    spin_netto_magnetic = H * magnetic_term
    return spin_netto_interaction + spin_netto_magnetic 

@njit
def metropolis_local(field,q,J = 1.7,H = 0,T = 2,kb = 1,count = 5e6, start_data_count = 0.5e6): # achtung niedrige temperatur und verhältnismäßig starkes feld 
    d1,d2 = np.shape(field)
    counter = 0 
    initial = energy(field, d1, d2, J, H)
    initial_energy = initial[0]
    energy_trend = [initial_energy]
    magnetic_initial = magnetization(field, d1, d2, q)
    magnetic_trend = [magnetic_initial[0]]
    magnetic_direction_trend = [magnetic_initial[1]]
    field_collection = []
    while counter <= count: 
        choiced1 = ran.randint(0,d1-1)
        choiced2 = ran.randint(0,d2-1)
        # start_spin = field[choiced1,choiced2]
        newfield = field.copy()
        newfield[choiced1,choiced2] = ran.randint(1,q)
        delta_E = energy_local(field,d1,d2,choiced1,choiced2,J,H) - energy_local(newfield,d1,d2,choiced1,choiced2,J,H)
        r = ran.uniform(0, 1)
        if delta_E > 0: 
             if r < np.exp(-delta_E/(T*kb)): 
                field = newfield
                field_collection.append(field)
                if counter >= start_data_count:
                    initial_energy += -delta_E
                    energy_trend.append(initial_energy)
                    magnetic_data = magnetization(field, d1, d2, q)
                    magnetic_trend.append(magnetic_data[0])
                    magnetic_direction_trend.append(magnetic_data[1])
        else: 
            field = newfield
            field_collection.append(field)
            if counter >= start_data_count:
                initial_energy += -delta_E
                energy_trend.append(initial_energy)
                magnetic_data = magnetization(field, d1, d2, q)
                magnetic_trend.append(magnetic_data[0])
                magnetic_direction_trend.append(magnetic_data[1])
        counter += 1
        # print(energy_trend[:100])
    return field, energy_trend, magnetic_trend, magnetic_direction_trend,field_collection

q_test = 2
test_field = spin_grid_generator((30,30),q_test)
# test_field = spin_grid_generator_cold((75,75))

data = metropolis_local(test_field, q_test)

show_field(data[0], q_test)
# fig = plt.figure(dpi=200)
fig,axs = plt.subplots(3, 1, constrained_layout=True)
axs[0].plot(data[1], '.',markersize=0.2)
axs[0].set_xlabel('iterations')
axs[0].set_title('Energy-Trend')
axs[0].set_ylabel('Energy')
axs[1].plot(data[2], '.',markersize=0.2)
axs[1].set_xlabel('iterations')
axs[1].set_title('Magnetic-Trend')
axs[1].set_ylabel('Magnetisation')
axs[2].plot(data[3], '.',markersize=0.2)
axs[2].set_xlabel('iterations')
axs[2].set_title('Magnetic-Direction-Trend')
axs[2].set_ylabel('Direction')

# def animation(field):
#     fig = plt.figure(dpi=200)
#     plt.title('Field-Evolution')
#     for i in range(len(field)):
#         show_field(field[3][i], q_test)
#         plt.pause(0.01)
   
# animation(data) 

heat_cap = np.var(data[1])
Xi = np.var(data[2])

print(heat_cap)
print(Xi)



