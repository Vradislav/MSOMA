import pandas as pd
import numpy as np
import random as rand
import math
from tqdm import tqdm
from more_itertools import sort_together 
from statistics import mean 

from itertools import *

import random
import time


class Msoma:
    
    def __init__(self, func):
        self.f = func
        
    # population creation (создание популяции)
    def pop_create(self, pop_size, num_of_x, a, b):
        pop = np.empty((num_of_x, pop_size))
        for j in range(pop_size):
            for i in range(num_of_x):
                pop[i][j] = a[i] + random.random() * (b[i] - a[i])
        return pop


    # sorting by cost function (сортировка по функции присособленности)
    def cost_function_sort(self, pop, pop_size, num_of_x):
        newpop = np.empty((num_of_x, pop_size))
        cost = {}
        for j in range(pop_size):
            cost[j] = self.f(pop[:, j])
        list_cost = list(cost.items())
        list_cost.sort(key=lambda i: i[1])
        for j in range(pop_size):
            newpop[:, j] = pop[:, list_cost[j][0]]
        return newpop


    # creating a PRT vector (создание вектора PRT)
    def create_prt_vector(self, prt, pop_size, num_of_x):
        prt_vector = np.zeros((num_of_x, pop_size))
        for j in range(pop_size):
            for i in range(num_of_x):
                if random.random() < prt:
                    prt_vector[i][j] = 1
        return prt_vector


    # movement of individuals towards leaders
    def movement(self, pop_size, prt, num_of_x, NStep, new_pop, a, b):

        vector_of_steps_1 = np.zeros((num_of_x, NStep * 4))
        vector_of_steps_2 = np.zeros((num_of_x, NStep * 2))
        vector_of_steps_3 = np.zeros((num_of_x, NStep))

        PRT = self.create_prt_vector(prt, pop_size * 3, num_of_x)

        # movement to the first leader
        for i in range(pop_size):
            for s in range(4 * NStep):
                vector_of_steps_1[:, s] = new_pop[:, i] + ((new_pop[:, 0] - new_pop[:, i])/(2*NStep)) * (PRT[:, i])*s
            new_pop[:, i] = self.cost_function_sort(vector_of_steps_1, 4 * NStep, num_of_x)[:, 0]

        # movement to the second leader
        for i in range(pop_size, pop_size * 2):
            for s in range(2 * NStep):
                vector_of_steps_2[:, s] = new_pop[:, i] + ((new_pop[:, pop_size+1] - new_pop[:, i])/NStep) * (PRT[:, i])*s
            new_pop[:, i] = self.cost_function_sort(vector_of_steps_2, 2 * NStep, num_of_x)[:, 0]

        # movement to the third leader
        for i in range(pop_size * 2, pop_size * 3):
            for s in range(NStep):
                vector_of_steps_3[:, s] = new_pop[:, i] + ((new_pop[:, 2*pop_size+2] - new_pop[:, i])/(NStep/2)) * (PRT[:, i])*s
            new_pop[:, i] = self.cost_function_sort(vector_of_steps_3, NStep, num_of_x)[:, 0]

        # catching value out of range
        for j in range(pop_size):
            for i in range(num_of_x):
                if new_pop[i][j] < a[i]:
                    new_pop[i][j] = a[i]
                if new_pop[i][j] > b[i]:
                    new_pop[i][j] = b[i]
        return new_pop


    # refinement function (функция уточнения)
    def elaboration(self, pop, NStep, prt, num_of_x):
        NStep *= 10
        PRT = self.create_prt_vector(prt, 3, num_of_x)
        vector_of_steps = np.zeros((num_of_x, NStep))
        ans = np.zeros((num_of_x, 3))
        for i in range(3):
            for s in range(NStep):
                vector_of_steps[:, s] = pop[:, i] + ((pop[:, 0] - pop[:, i])/(NStep/2)) * (PRT[:, i])*s
            ans[:, i] = self.cost_function_sort(vector_of_steps, NStep, num_of_x)[:, 0]
        ans = self.cost_function_sort(ans, 3, num_of_x)
        return ans[:, 0]


    # MSOMA find min
    def find_min(self
                 , pop_size = 100
                 , num_of_x = 2
                 , a = [-100, -100]
                 , b = [100, 100]
                 , NStep = 20
                 , prt = 0.3
                 , Migrations = 25
                 , MinDist = 0.1**10):
        start_time = time.time()

        pop = self.pop_create(pop_size, num_of_x, a, b)
        new_pop = self.cost_function_sort(pop, pop_size, num_of_x)

        # cloning
        new1_pop = np.concatenate((new_pop, new_pop), axis=1)
        new_pop = np.concatenate((new1_pop, new_pop), axis=1)

        MCount = 0
        # migration loop
        pbar = tqdm(total=Migrations)
        
        while MCount < Migrations and math.sqrt((1/2) * ((self.f(new_pop[:, 1]) - self.f(new_pop[:, 0]))**2 + (self.f(new_pop[:, 2]) - self.f(new_pop[:, 0]))**2)) >= MinDist:
            new_pop = self.movement(pop_size, prt, num_of_x, NStep, new_pop, a, b)
            new_pop = self.cost_function_sort(new_pop, 3 * pop_size, num_of_x)

            # population refreshing
            new_pop = new_pop[:, :round(pop_size*(2/3))]
            new_pop = np.concatenate((new_pop, self.pop_create(round(pop_size*(1/3)), num_of_x, a, b)), axis=1)
            new_pop = self.cost_function_sort(new_pop, pop_size, num_of_x)

            # cloning
            new1_pop = np.concatenate((new_pop, new_pop), axis=1)
            new_pop = np.concatenate((new1_pop, new_pop), axis=1)

            MCount += 1
            pbar.update(1)
            
        pbar.close()
        last_pop = new_pop[:, :3]

        ans = self.elaboration(last_pop, NStep, prt, num_of_x)

        return ans, self.f(ans), MCount, time.time() - start_time