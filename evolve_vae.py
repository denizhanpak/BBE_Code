import logging

import numpy as np
import gym
import pandas as pd

import hydra
from hydra.core.config_store import ConfigStore

import pytorch_lightning as pl
from omegaconf import OmegaConf

from chickai.algo import PPO
from chickai.algo import EvolutionarySearch
from chickai.common.config import Config
from chickai.env import ChickAIEnv, PytorchVisualEnv
from chickai.raycast_env import CTRNNEnv, VAEEnv
from chickai.agent import CTRNNAgent

logger = logging.getLogger(__name__)

def _absolutize_paths(config):
    # Absolutize paths.
    if config.env.env_path is not None:
        config.env.env_path = hydra.utils.to_absolute_path(
            config.env.env_path)

    if config.env.imprint_video is not None:
        config.env.imprint_video = hydra.utils.to_absolute_path(
            config.env.imprint_video)

    if config.env.test_video is not None:
        config.env.test_video = hydra.utils.to_absolute_path(
            config.env.test_video)

    return config


cs = ConfigStore.instance()
cs.store(name="config", node=Config)

def get_bounds(neuron_count=10, genome_len=61):
    lb = []
    ub = []

    tau_min = 0.01
    tau_max = 5.0
    bias_min = -15
    bias_max = 15
    weight_min = -15
    weight_max = 15

    lb += ([tau_min] * neuron_count)
    ub += ([tau_max] * neuron_count)
    
    lb += ([bias_min] * neuron_count)
    ub += ([bias_max] * neuron_count)
    
    weight_count = genome_len - (neuron_count * 2)
    lb += ([weight_min] * weight_count)
    ub += ([weight_max] * weight_count)

    return lb, ub


def get_initial(pop_size):

    #PARAMS for best RAY CAST Agent
    return np.array([ 4.13897649e+00, 4.03643237e+00, 1.40668414e+00, 4.46684594e+00,
  1.51635290e+00,-2.85244578e+00, 1.79864592e+00,-1.14560575e+01,
  9.91549802e+00, 6.46746720e+00, 4.15080560e+00, 4.23103439e+00,
 -6.29444275e+00,-3.19445423e-01,-1.20578810e+01,-7.88906879e-01,
 -7.11187518e+00, 1.54076675e-01,-1.26418151e+01,-2.40271655e+00,
  1.43192743e+01, 1.11855197e+01,-3.48095603e+00, 1.36722615e+01,
  1.90341647e+00, 2.99302443e+00, 4.38836003e+00,-6.40687534e-03,
  1.06454073e+01,-7.99431166e+00,-9.81262564e+00,-4.76225122e-01,
 -7.92729064e+00,-7.69626320e+00,-1.06106531e+01, 2.73192323e+00,
 -1.07922743e+01, 3.58344396e+00, 1.78477359e+00,-7.47680318e+00,
 -7.32674007e+00, 1.08000867e+01,-1.11531948e+01, 6.27124276e+00,
 -3.52103606e+00, 3.24680276e+00])

@hydra.main(config_path="configs", config_name="config")
def train(config: Config):
    logger.info("Configuration\n" + OmegaConf.to_yaml(config))
    config = _absolutize_paths(config)

    pop_size = 100
    its = 70
    sensor_count = 5
    inter_count = 3
    motor_count = 2
    tmp = CTRNNAgent(motor_count=motor_count,sensor_count=sensor_count,inter_count=inter_count)

    lb, ub = tmp.get_range()
    genome_size = tmp.param_count()

    env = VAEEnv(CTRNNEnv(**config.env, use_visual=True))
    #env = RaycastEnv(CTRNNEnv(**config.env, use_visual=False))
    
    #return test(env)

    algo = EvolutionarySearch(env,pop_size=pop_size, genotype_size=genome_size, lb=lb, ub=ub, max_iter=its)
    best = algo.run()
    logger.info("Best is :")
    logger.info(best)

def test(env):
    #TestAgent(env)
    
    
    total_score = 0
    episode_num = 10
    max_step = 500
    agent = CTRNNAgent()
    agent.set_weights(get_initial(0))
    counter = 0
    total = 100
    for i in range(episode_num):
        obs = env.reset()
        agent.brain.randomize_outputs(0.5,0.51)
        for i in range(max_step):
            action = agent.Act(obs)
            #action = np.array([1,0])
            #print("Action:", action)
            print("Observation", obs)

            if np.isnan(action).any():
                return 10

            obs, reward, done, info = env.step(action)
            if done:
                total_score += reward
                #print("Done")
                #print(reward)
                break

    return

if __name__ == "__main__":
    #print(get_initial(1))
    data = train()
    print(data)
