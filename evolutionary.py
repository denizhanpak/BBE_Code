'''
Evolutionary search for evolutionary robotics style research
Will assume all genotypes are converted to CTRNN
Genotypes have the following structure:
    - [0:N] taus for N neruons
    - [N:2N] Biases for N neurons
    - [(2+i)*N:(3+i)*N] for input weights of each of N neurons
Code taken from Demo of EvolSearch (By Madhavun Candadai)
'''

import numpy as np
import matplotlib.pyplot as plt
from chickai.agent import CTRNNAgent
from sko.GA import GA
import pandas as pd

def EvaluateGenome(genome):
    episode_num = 6
    max_step = 500
    global E
    env = E
    agent = CTRNNAgent()
    agent.set_weights(genome)
    total_score = 0
    for i in range(episode_num):
        obs = env.reset()
        agent.brain.randomize_outputs(0.5,0.51)
        for i in range(max_step):
            action = agent.Act(obs)
            
            if np.isnan(action).any():
                return 10

            obs, reward, done, info = env.step(action)
            if done:
                total_score += reward
                break

    return -total_score

def TestAgent(env):
    genome = np.ones(3+3+6)
    print(EvaluateGenome(genome, env))



def Agent_Evaluation(individual):
    '''
    convert genome to ctrnn
    This will connect to a unity environment
    run environment with ctrnn
    '''
    current = multiprocessing.current_process()
    print(current.pid)
    return np.mean(individual)

class EvolutionarySearch:
    def __init__(self, env, pop_size=100, genotype_size=61, lb=-1, ub=1, max_iter=100, init=None):
        
        self.env = env
        global E 
        E = env
        
        #The evolutionary search object
        self.ga = GA(func=EvaluateGenome, n_dim=genotype_size, size_pop=pop_size, max_iter=max_iter, lb=lb, ub=ub, precision=1e-7)
        if init is not None:
            self.ga.Chrom = init

    def calc_genotype(self, pheno_dict):
        genotype_length = 0
        for key in pheno_dict:
            genotype_length += len(pheno_dict[key])
        return genotype_length

    def run(self,increments=10):
        
        total_its = self.ga.max_iter // increments
        for i in range(total_its):
            best_x, best_y = self.ga.run(max_iter=increments)
            gen = (i+1) * increments
            print(f'Generation: {gen}')
            print('best_x:', best_x, '\n', 'best_y:', best_y)
            # %% Plot the result

        Y_history = pd.DataFrame(self.ga.all_history_Y)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.show()
        fig.savefig('./evo_run.pdf')
        Y_history.to_pickle("Run_history.pkl")

        return best_x

