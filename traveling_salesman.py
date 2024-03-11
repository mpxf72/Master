import numpy as np
import matplotlib.pyplot as plt


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
N = 50

beta_min = 1
beta_max = 200
deltabeta = 1
beta_power = 1.3
beta = beta(beta_min, beta_max, deltabeta, beta_power)

N_s = 1000
M = N_s * len(beta)
print("total iterations are M = %g" %M)


### minimizing a specific path ###
# city_pos, dis_mat = generate_map(dim, N)
# p, d_total = simulated_annealing(N, dis_mat, beta, N_s, True, city_pos, 5)

### calculating average with error ###
n_runs = 30
dataset = generating_data(dim, N, beta, n_runs, N_s)
D_average, sigma = jackknife(dataset)
print("average minimal path lenght is %.3f ± %.3f" %(D_average, sigma))


