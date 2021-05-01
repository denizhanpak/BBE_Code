# imports
import numpy as np
import matplotlib.pyplot as plt
# importing the CTRNN class
from CTRNN import CTRNN
from scipy.sparse import csr_matrix

def force_range(x, lb, ub):
    return np.maximum(np.minimum(x, ub), lb)


# Assumes the structure that 
class CTRNNAgent:
    def __init__(self, genome=None, motor_count=2, sensor_count=5, inter_count=3, step_size=0.1):
        self.neuron_counts = [inter_count, motor_count]
        self.step_size = step_size
        self.size = sum(self.neuron_counts)
        self.sensor_count = sensor_count
        self.inter_range = range(inter_count)
        self.motor_range = range(inter_count,self.size)
        self.brain = CTRNN(size=self.size,step_size=step_size)
        self.brain.randomize_outputs(0.1,0.9)
        if genome != None: 
            self.set_weights(genome)

    def get_range(self):
        tau_min = 0.01
        tau_max = 5.0
        bias_range = 15
        weight_range = 15

        ub = []
        lb = []
        
        ub += [tau_max] * self.size
        lb += [tau_min] * self.size
        
        ub += [bias_range] * self.size
        lb += [-bias_range] * self.size
        
        total = 0
        total += self.sensor_count * len(self.inter_range) #sensor to inter weight count
        total += len(self.inter_range) ** 2 #inter to inter weight count
        total += len(self.inter_range) * len(self.motor_range) #inter to motor weight count
        total += len(self.motor_range) * len(self.inter_range) #motor to inter weight count

        ub += [weight_range] * total
        lb += [-weight_range] * total
        
        return lb, ub
    
    def set_weights(self, genome):
        
        g_index = 0

        tau_array = []
        for i in range(self.size):
            tau_array.append(genome[g_index])
            g_index += 1
        self.brain.taus = np.array(tau_array)

        
        bias_array = []
        for i in range(self.size):
            bias_array.append(genome[g_index])
            g_index += 1
        self.brain.bias = np.array(bias_array)
        
        self.sensor_weights = []
        for i in range(self.sensor_count):
            self.sensor_weights.append([])
            for _ in self.inter_range:
                self.sensor_weights[i].append(genome[g_index])
                g_index += 1


        weight_matrix = np.zeros((self.size,self.size))
        
        for i in self.inter_range:
            for j in self.inter_range:
                weight_matrix[i,j] = genome[g_index]
                g_index += 1
   
        for i in self.inter_range:
            for j in self.motor_range:
                weight_matrix[i,j] = genome[g_index]
                g_index += 1
   
        for i in self.motor_range:
            for j in self.inter_range:
                weight_matrix[i,j] = genome[g_index]
                g_index += 1

        self.brain.weights = csr_matrix(weight_matrix)
        


    def Act(self, observation: np.array, return_trajectory=False):
        rv = []

        obs = []
        for j in self.inter_range:
            o = 0
            for i in range(self.sensor_count):
                o += self.sensor_weights[i][j] * observation[i]
            obs.append(o)

        obs[1] = 0

        N = self.size - len(obs)
        observation = np.pad(obs, (0, N), 'constant')

        steps = int(1 / self.step_size)

        rv.append(self.brain.states)
        self.brain.euler_step(observation)
        rv.append(self.brain.states)
        
        for i in range(steps - 1):
            self.brain.euler_step(np.zeros(self.size))
            rv.append(self.brain.states)
            
        
        outputs = len(self.motor_range)

        if return_trajectory:
            return self.brain.outputs[-outputs:], rv
        return self.brain.outputs[-outputs:]

    def get_mapping(self):
        rv = {}
        
        rv["taus"] = [0] * self.size
        rv["biases"] = [0] * self.size
        rv["sensor_inter"] = [0] * (self.sensor_count * len(self.inter_range))
        rv["inter_inter"] = [0] * (len(self.inter_range) ** 2)
        rv["inter_motor"] = [0] * (len(self.inter_range) * len(self.motor_range))
        rv["motor_inter"] = [0] * (len(self.inter_range) * len(self.motor_range))

        return rv

    def param_count(self):
        total = 0
        total += self.size #tau count
        total += self.size #bias count
        total += self.sensor_count * len(self.inter_range) #sensor to inter weight count
        total += len(self.inter_range) ** 2 #inter to inter weight count
        total += len(self.inter_range) * len(self.motor_range) #inter to motor weight count
        total += len(self.motor_range) * len(self.inter_range) #motor to inter weight count
        return total

    def __repr__(self):
        s = "Sensor Neurons:\n" 

        for i in self.inter_range:
            s += f"dy_{i}/dt = 1/{self.brain.taus[i]} (-y_{i} + "
            for j in self.inter_range:
                w = self.brain.weights[i,j]
                if w != 0:
                    s += f"{w} SIG(y_{j} + {self.brain.bias[j]}) + "
            for j in self.motor_range:
                w = self.brain.weights[i,j]
                if w != 0:
                    s += f"{w} SIG(y_{j} + {self.brain.bias[j]}) + "
            for j in range(self.sensor_count):
                s += f"{self.sensor_weights[j][i]} S_{j} + "
            s = s[:-2]
            s += "\n"
        
        s += "Motor Neurons:\n"
        for i in self.motor_range:
            s += f"dy_{i}/dt = 1/{self.brain.taus[i]} (-y_{i} + "
            for j in self.inter_range:
                w = self.brain.weights[i,j]
                if w != 0:
                    s += f"{w} SIG(y_{j} + {self.brain.bias[j]}) + "
            s = s[:-2]
            s += "\n"
        

        return s
        

#tmp = CTRNNAgent(inter_count=3, sensor_count=5, motor_count=2)
#tmp.set_weights(np.ones(46))
#print(tmp)
