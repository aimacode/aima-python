import networkx as nx
import matplotlib.pyplot as plt
import sys
import numpy as np
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from search import *
from matplotlib.pyplot import pause
from random import shuffle


class TSP_problem(Problem):

    '''
    subclass of Problem to define various functions 
    '''

    def two_opt(self, state):
        '''
        Neighbour generating function for Traveling Salesman Problem
        '''
        state2 = state[:]
        l = random.randint(0, len(state2) - 1)
        r = random.randint(0, len(state2) - 1)
        if l > r:
            l, r = r,l
        state2[l : r + 1] = reversed(state2[l : r + 1])
        return state2

    def actions(self, state):
        '''
        action that can be excuted in given state
        '''
        return [self.two_opt]
    
    def result(self, state, action):
        '''
        result after applying the given action on the given state
        '''
        return action(state)

    def path_cost(self, c, state1, action, state2):
        '''
        total distance for the Traveling Salesman to be covered if in state2
        '''
        cost = 0
        for i in range(len(state2) - 1):
            cost += distances[state2[i]][state2[i + 1]]
        cost += distances[state2[0]][state2[-1]]
        return cost
 
    def value(self, state):
        '''
        value of path cost given negative for the given state
        '''
        return -1 * self.path_cost(None, None, None, state)


def path_cost(state):
    '''
    Path cost function based on given state
    '''
    cost = 0
    for i in range(len(state) - 1):
        cost += distances[state[i]][state[i + 1]]
    cost += distances[state[0]][state[-1]]
    return cost


font = {'family': 'roboto',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }

cities = []
distances ={}

def simulated_annealing2(problem, iteration, schedule=exp_schedule()):
    edges = []
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        T = schedule(t)


        G.remove_edges_from(edges)
        edges = []
        for i in range(len(current.state)-1):
            edges.append((current.state[i], current.state[i + 1]))
        edges.append((current.state[0], current.state[-1]))
        G.add_edges_from(edges)
        plt.clf()
        nx.draw(G, romania_map.locations, node_shape = 'h', color = 'g', with_labels = True)
        plt.text(450, 580, "Cost = " + str('{:.2f}'.format(path_cost(current.state))), fontdict=font)
        plt.text(450, 565, "Temperature = " + str('{:.2f}'.format(T)), fontdict=font)
        plt.text(450, 550, "Iteration Number = " + str(iteration+1), fontdict=font)
        plt.pause(0.05)

        if T == 0:
            return current.state
        neighbors = current.expand(problem)
        if not neighbors:
            return current.state
        next = random.choice(neighbors)
        delta_e = problem.value(next.state) - problem.value(current.state)
        if delta_e > 0 or probability(math.exp(delta_e / T)):
            current = next



if __name__ == "__main__":

    # creating initial path
    for name in romania_map.locations.keys():    
        distances[name] = {}
        cities.append(name)

    shuffle(cities)

    # distances['city1']['city2'] contains euclidean distance between their coordinates
    for name_1,coordinates_1 in romania_map.locations.items():
        for name_2,coordinates_2 in romania_map.locations.items():
            distances[name_1][name_2] = np.linalg.norm([coordinates_1[0] - coordinates_2[0], coordinates_1[1] - coordinates_2[1]])
            distances[name_2][name_1] = np.linalg.norm([coordinates_1[0] - coordinates_2[0], coordinates_1[1] - coordinates_2[1]])

    # Initialize networkx graph
    G = nx.Graph()
    plt.gcf()
    tsp_problem = cities
    for i in range(100):
        tsp_problem = TSP_problem(tsp_problem)
        tsp_problem = simulated_annealing2(tsp_problem, i)