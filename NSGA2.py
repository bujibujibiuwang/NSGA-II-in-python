import random
import numpy as np
import matplotlib.pyplot as plt


def fast_non_dominated_sort(values1, values2):
    size = len(values1)
    dominate_set = [[] for _ in range(size)]  # 解p支配的解集合
    dominated_count = [0 for _ in range(size)]  # 支配p的解数量
    solution_rank = [0 for _ in range(size)]  # 每个解的等级
    fronts = [[]]

    for p in range(size):
        dominate_set[p] = []
        dominated_count[p] = 0
        for q in range(size):
            if values1[p] <= values1[q] and values2[p] <= values2[q] \
                    and ((values1[p] == values1[q]) + (values2[p] == values2[q])) != 2:
                dominate_set[p].append(q)
            elif values1[q] <= values1[p] and values2[q] <= values2[p] \
                    and ((values1[q] == values1[p]) + (values2[q] == values2[p])) != 2:
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            solution_rank[p] = 0
            fronts[0].append(p)

    level = 0
    while fronts[level]:
        Q = []
        for p in fronts[level]:
            for q in dominate_set[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    solution_rank[q] = level + 1
                    if q not in Q:
                        Q.append(q)
        level = level + 1
        fronts.append(Q)
    del fronts[-1]
    return fronts


def crowed_distance_assignment(values1, values2, front):
    length = len(front)
    sorted_front1 = sorted(front, key=lambda x: values1[x])
    sorted_front2 = sorted(front, key=lambda x: values2[x])
    dis_table = {sorted_front1[0]: np.inf, sorted_front1[-1]: np.inf, sorted_front2[0]: np.inf, sorted_front2[-1]: np.inf}
    for i in range(1, length - 1):
        k = sorted_front1[i]
        dis_table[k] = dis_table.get(k, 0)+(values1[sorted_front1[i+1]]-values1[sorted_front1[i-1]])/(max(values1)-min(values1))
    for i in range(1, length - 1):
        k = sorted_front1[i]
        dis_table[k] = dis_table[k]+(values2[sorted_front2[i+1]]-values2[sorted_front2[i-1]])/(max(values2)-min(values2))
    distance = [dis_table[a] for a in front]
    return distance



def function1(x):
    y = x ** 2
    return round(y, 2)


def function2(x):
    y = (x - 2) ** 2
    return round(y, 2)


def crossover(x, y):
    r = random.random()
    if r > 0.5:
        return mutation((x+y)/2)
    else:
        return mutation((x-y)/2)


def mutation(solution):
    min_v = -55
    max_v = 55
    mutation_prob = random.random()
    if mutation_prob < 1:
        solution = min_v + (max_v - min_v) * random.random()
    return solution


def main_loop(pop_size, max_gen, init_population):
    gen_no = 0
    population_P = init_population.copy()
    while gen_no < max_gen:
        population_R = population_P.copy()
        # 根据P(t)生成Q(t),R(t)=P(t)vQ(t)
        while len(population_R) != 2 * pop_size:
            x = random.randint(0, pop_size - 1)
            y = random.randint(0, pop_size - 1)
            population_R.append(crossover(population_P[x], population_P[y]))
        # 对R(t)计算非支配前沿
        objective1 = [function1(population_R[i]) for i in range(2 * pop_size)]
        objective2 = [function2(population_R[i]) for i in range(2 * pop_size)]
        fronts = fast_non_dominated_sort(objective1, objective2)
        # 获取P(t+1)，先从等级高的fronts复制，然后在同一层front根据拥挤距离选择
        population_P_next = []
        choose_solution = []
        level = 0
        while len(population_P_next) + len(fronts[level]) <= pop_size:
            for s in fronts[level]:
                choose_solution.append(s)
                population_P_next.append(population_R[s])
            level += 1
        if len(population_P_next) != pop_size:
            level_distance = crowed_distance_assignment(objective1, objective2, fronts[level])
            sort_solution = sorted(fronts[level], key=lambda x: level_distance[fronts[level].index(x)], reverse=True)
            for i in range(pop_size - len(population_P_next)):
                choose_solution.append(sort_solution[i])
                population_P_next.append(population_R[sort_solution[i]])
        # 得到P(t+1)重复上述过程
        population_P = population_P_next.copy()
        if gen_no % 50 == 0:
            best_obj1 = [function1(population_P[i]) for i in range(pop_size)]
            best_obj2 = [function2(population_P[i]) for i in range(pop_size)]
            f = fast_non_dominated_sort(best_obj1, best_obj2)
            print(f'generation {gen_no}, first front:')
            for s in f[0]:
                print(round(population_P[s], 2), end=' ')
            print('\n')
        gen_no += 1
    return best_obj1, best_obj2


min_x = -55
max_x = 55
pop_size = 40
max_gen = 900
population_P = [min_x+(max_x-min_x)*random.random() for i in range(0, pop_size)]
v1, v2 = main_loop(pop_size, max_gen, population_P)
plt.scatter(v1, v2)
plt.show()
