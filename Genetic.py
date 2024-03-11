#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:40:43 2024

@author: maximilianpfandner
"""
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import numba

# @jit
def city_maker(N, q_grid):
    ''' due to very small probability for double choice negelctable if statement'''
    space = np.linspace(0,1,q_grid)
    cities = []
    for _ in range(N): 
        x1 = np.random.choice(space)
        x2 = np.random.choice(space)
        cities.append([x1,x2])
    return cities

def dist_mat(city_list,N): 
    D = np.zeros((N,N))
    # D = []
    # for i in range(len(city_list)): 
    #     D.append([])
    #     print(D)
    # print(D)
    for i in range(len(city_list)): 
        for j in range(len(city_list)): 
                dist = np.sqrt((city_list[i][0] - city_list[j][0])**2 + (city_list[i][1] - city_list[j][1])**2)
                # D[i].append(dist)
                D[i,j] = dist
    return D 
# @jit
def get_distance(D_M, permutation): 
    d = 0
    for i in range(len(permutation)-1): 
        d += D_M[permutation[i],permutation[i + 1]]
    return d 
        
# @jit
def genetic_code(D_M, mu_rate = 0.6270370370370371, ma_rate = 0.5907407407407407, kill_rate = 0.8, inc_rate = 1, start_pool = 50000, sweeps = 10000, mutation = 'True', mating = 'True'): 
    permutations = []
    N = np.shape(D_M)[0]
    N_list = []
    distance_end = []
    for i in range(N): 
        N_list.append(i)
    for _ in range(start_pool): 
        permutation = []
        N_list_new = N_list.copy()
        for i in range(len(N_list)):
            a = np.random.choice(N_list_new)
            permutation.append(a)
            N_list_new.remove(permutation[-1])
        permutation.append(permutation[0]) 
        permutations.append(permutation)
    for step in range(sweeps): 
        distance = []
        start_pool_ = len(permutations)
        if mutation == 'True': 
            for i in range(round(start_pool*mu_rate)): # mutation rate % 
                ind_1 = np.random.randint(0,start_pool_ - 1)
                N_list_new = N_list.copy()
                N_list_new.remove(permutations[ind_1][0]) # end and start point doesn't change 
                ind_2 = np.where(permutations[ind_1] == np.random.choice(N_list_new))[0][0]
                ind_3 = np.where(permutations[ind_1] == np.random.choice(N_list_new))[0][0]
                permutations[ind_1][ind_2], permutations[ind_1][ind_3] = permutations[ind_1][ind_3], permutations[ind_1][ind_2]
        if mating == 'True':
            for i in range(round(start_pool*ma_rate)): # mating rate  %
                ind_1 = np.random.randint(0,start_pool_-1)
                child = permutations[ind_1][0:round(N/2)] + permutations[ind_1][round(N/2):]
                if child[0] == child[-1]: 
                    counting = []
                    for i in child: 
                        counting.append(child.count(i))
                    if sum(counting) <= N: 
                        permutations.append(child)
        for p in permutations:
            d = get_distance(D_M,p)
            distance.append(d)
        max_ = max(distance)
        min_ = min(distance)
        count = 0
        G = round(len(distance)*inc_rate) 
        while len(distance) < G: 
                i = np.random.choice(len(distance)-1)
                if distance[i] >= max_ - (max_-min_)*kill_rate: # kill_rate % with least fitness
                    permutations.remove(permutations[i])
                    distance.remove(distance[i])
    for p in permutations:
        d = get_distance(D_M,p)
        distance_end.append(d)
    p_end = np.where(distance_end == min(distance_end))
    P = permutations[p_end[0][0]]
    return P, min(distance_end)

N = 20
q_grid = 100
city_list = city_maker(N, q_grid)   
D_M = dist_mat(city_list,N)  
permutation, distance = genetic_code(D_M)
print(sorted(permutation),permutation)

def plotter(cities, permutation, distance): 
    d = distance
    x_ = []
    y_ = []
    for i in range(len(permutation)-1): 
            label = f"{i}"
            x1 = cities[permutation[i]][0]
            y1 = cities[permutation[i]][1]
            x2 = cities[i][0]
            y2 = cities[i][1]
            plt.annotate(label, # this is the text
                          (x2,y2), # these are the coordinates to position the label
                          textcoords="offset points", # how to position the text
                          xytext=(0,10), # distance from text to points (x,y)
                          ha='center')
            plt.xlim((0,1.1))
            plt.ylim((0,1.1))
            x_.append(x1)
            y_.append(y1)
            # x_dis = cities[permutation[i]][0] - cities[permutation[i + 1]][0]
            # y_dis = cities[permutation[i]][1] - cities[permutation[i + 1]][1]
            # x_space = np.linspace(0,x_dis,100)
            # y_space = cities[permutation[i]][1] + x_space * (y_dis/x_dis)
            # x_space_new = cities[permutation[i]][0] + x_space
    # print(cities, x_, y_ )
    plt.plot(x_,y_, 'o--')
    plt.xlim(-0.1,1.2)
    plt.ylim(-0.1,1.2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('total distance = %f' % d)
    
plotter(city_list, permutation, distance)

def get_optimum(D_M, mu_rate, ma_rate): 
    M = np.zeros((len(ma_rate),len(ma_rate)))
    for i in range(len(mu_rate)):
        for j in range(len(ma_rate)): 
            M[i,j] = genetic_code(D_M,i,j)[1]
    mini = np.min(M)
    M = np.where(M == mini)
    x = M[0][0]
    y = M[1][0]
    return mini, mu_rate[x], ma_rate[y]
            
            
m1 = np.linspace(0.01,0.99,30)  
n1 = np.linspace(0.01,0.99,30)  
# print(get_optimum(D_M, m1, n1))
n_list = [20,40]

# for n in n_list: 
#     q_grid = 100
#     city_list = city_maker(n, q_grid)  
#     DI = []
#     MU = []
#     MA = []
#     for _ in range(3): 
#         di, mu_rate, ma_rate = get_optimum(D_M, m1, n1)
#         DI.append(di)
#         MU.append(mu_rate)
#         MA.append(ma_rate)
#     av_di = np.sum(DI)/3
#     av_mu = np.sum(MU)/3
#     av_ma = np.sum(MA)/3
#     print('N=',n,av_di,av_mu,av_ma)
        
    
    
    
    






########################################################################################
# vergleich daniel 


def generate_map(dim, N):
    city_pos = np.zeros((N, dim))
    for n in range(N):
        for d in range(dim):
            city_pos[n, d] = np.random.rand()
    dis_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i < j:
                d_vec = np.zeros((dim))
                for d in range(dim):
                    d_vec[d] = city_pos[i, d] - city_pos[j, d]
                dis_mat[i, j] = np.linalg.norm(d_vec)
                dis_mat[j, i] = dis_mat[i, j]

    return city_pos, dis_mat


def plot_map(city_pos, p, d_total, beta):
    new_city_pos = np.zeros_like(city_pos)
    for i, pi in enumerate(p):
        new_city_pos[i, 0] = city_pos[pi, 0]
        new_city_pos[i, 1] = city_pos[pi, 1]
    fig = plt.figure(dpi=150)
    plt.plot(new_city_pos[:, 0], new_city_pos[:, 1], ".--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("total distance = %.4f, β = %g" %(d_total, beta))
    xs = city_pos[:, 0]
    ys = city_pos[:, 1]
    for n in range(len(xs)):
        label = f"{n}"
        x = xs[n]
        y = ys[n]
        plt.annotate(label, # this is the text
                      (x,y), # these are the coordinates to position the label
                      textcoords="offset points", # how to position the text
                      xytext=(0,10), # distance from text to points (x,y)
                      ha='center')
    plt.xlim((0,1.1))
    plt.ylim((0,1.1))


def total_distance(p, dis_mat): 
    distance = 0
    for i in range(len(p)):
        distance += dis_mat[p[i], p[(i + 1)%len(p)]]
    return distance


def swap_two(l, pos1, pos2):
    l_copy = l.copy()
    l_copy[pos1], l_copy[pos2] = l_copy[pos2], l_copy[pos1]
    return l_copy

def swap_inverse(l, pos1, pos2):
    l_copy = l.copy()
    l_copy[pos1 : pos2+1] = reversed(l_copy[pos1 : pos2+1])
    return l_copy

def swap_bunch(l, pos1, pos2):
    list0 = l[0:pos1]
    list1 = l[pos1:pos2+1]
    list2 = l[pos2+1:]
    new_list = list0 + list2
    p = np.random.randint(0, len(new_list)+1, dtype=int)
    for i, n in enumerate(list1):
        new_list.insert(p+i, n)
    return new_list


def simulated_annealing(N, dis_mat, beta, N_s, plotting, city_pos, n_plots):
    p = [i for i in range(N)]
    d_total = total_distance(p, dis_mat)
    plot_k = int(len(beta)/n_plots)
    k = plot_k
    for b in beta:
        for i in range(N_s):
            pro = np.random.randint(0, 3, dtype=int)
            # pro = 2
            if pro == 0:
                pos1 = np.random.randint(0, N, dtype=int)
                pos2 = np.random.randint(pos1, N, dtype=int)
                p_prop = swap_bunch(p, pos1, pos2)
                d_prop = total_distance(p_prop, dis_mat)
            elif pro == 1:
                pos1 = np.random.randint(0, N, dtype=int)
                pos2 = np.random.randint(pos1, N, dtype=int)
                p_prop = swap_inverse(p, pos1, pos2)
                d_prop = total_distance(p_prop, dis_mat)
            else:
                pos1 = np.random.randint(0, N, dtype=int)
                pos2 = np.random.randint(0, N, dtype=int)
                p_prop = swap_two(p, pos1, pos2)
                d_prop = total_distance(p_prop, dis_mat)
            if d_prop < d_total:
                p = p_prop
                d_total = d_prop
            else:
                r = np.random.uniform(0, 1)
                if r < np.exp(-(d_prop - d_total)*b):
                    p = p_prop
                    d_total = d_prop
        if k == plot_k and plotting or plotting and b == beta[-1]:
            plot_map(city_pos, p, d_total, b)
            k = 0
        else:
            k += 1
    return p, d_total


def beta(beta_min, beta_max, deltabeta, beta_power):
    beta = [beta_min]
    n = 1
    while True:
        if beta[-1] > beta_max or beta_min == beta_max:
            break
        beta.append(beta_min + (n*deltabeta)**beta_power)
        n += 1
    return beta


def jackknife(dataset):
    t = len(dataset)
    raw_avrg = np.average(dataset)
    avrg_n = np.zeros(t)
    sum_1 = 0
    for n in range(t):
        avrg_n[n] = np.average(np.delete(dataset, n))
        sum_1 += (avrg_n[n] - raw_avrg)**2
    sigma = np.sqrt((t - 1)/t * sum_1)
    avrg_bias = 1/t * np.sum(avrg_n)
    true_avrg = raw_avrg - (t - 1)*(avrg_bias - raw_avrg)
    return true_avrg, sigma


def generating_data(dim, N, beta, n_runs, N_s):
    D_total = np.zeros(n_runs)
    for u in range(n_runs):
        city_pos, dis_mat = generate_map(dim, N)
        p, d_total = simulated_annealing(N, dis_mat, beta, N_s, False, city_pos, 5)
        D_total[u] = d_total
        percentage = 100/n_runs * (u+1)
        print(percentage, "%")
    return D_total

dim = 2
N = 10

beta_min = 1
beta_max = 200
deltabeta = 1
beta_power = 1.3
beta = beta(beta_min, beta_max, deltabeta, beta_power)

N_s = 1000
M = N_s * len(beta)
print("total iterations are M = %g" %M)


## minimizing a specific path ###
city_pos, dis_mat = np.asanyarray(city_list), D_M
p, d_total = simulated_annealing(N, dis_mat, beta, N_s, True, city_pos, 5)

### calculating average with error ###
# n_runs = 30
# dataset = generating_data(dim, N, beta, n_runs, N_s)
# D_average, sigma = jackknife(dataset)
# print("average minimal path lenght is %.3f ± %.3f" %(D_average, sigma))
            
        
    
    
                
                
    
   