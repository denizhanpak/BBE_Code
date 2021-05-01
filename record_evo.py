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
from chickai.raycast_env import CTRNNEnv, RaycastEnv
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
    np.array([  3.49435282,  2.13126405,  1.37062444,  2.06295997,  2.33758512,
   4.63290504,  0.94449353,  4.36445864,  4.19807981,  3.57137925,
   2.68645664,  1.17515701, -5.04722752, -5.07231819, -7.60286348,
  11.42816902, 11.88946969,-14.53619415,  1.58662451,-10.02875994,
   0.5265841 , 13.9538431 ,  8.12016225, 11.17171844, 13.46626421,
 -11.27396313,  9.2766478 ,-11.75947118, -5.8074748 , -8.81268537,
  13.02977079,-10.06412883, -9.5135657 , -2.58755981, -6.94939092,
  -3.25665826,  0.24396907,-10.18266263, 14.44939317, -6.18729392,
  -6.01220356, -9.98496975,  1.61183917, -1.50078136,  3.09413282,
   7.17808752,  2.53514286, 14.24101883,  0.88935762,  9.33069124,
  -9.52964001,-10.72497596,-10.31713731, -7.93083735, 14.30819157,
  -8.34706071,  0.2559016 ,  6.47461461, -7.44504784,  6.9950272,
  10.75145993, -5.24527954,-12.79298275,  5.70571313,-11.22257949,
   8.71995018,  6.76803565, -1.95129206, 10.42190758,  5.33456391,
   8.4703243 , 12.70451629, -7.08171569, -4.51108903, -5.93455104,
   2.09081666,  4.87136546,-10.02400259,  4.55596395, -8.02028992,
  -8.10964352, -5.83386623, 10.6343862 , -5.06574962, -4.16456701,
  -7.13818005, -8.05255958, 10.6332766 , -1.37974159,-11.71066839,
 -14.27053101,-14.87759544, -6.25498757, -6.76531136, 10.66185107,
  -5.33893865,  3.35125792,  3.72266575,  8.08553683])

    np.array([  2.46052523,  3.41029172,  1.44862115,  0.39181211,  1.95604724,
   2.82504514,  3.75382212,  0.50058658,  2.5067388 ,  2.24265841,
   4.79427936,  0.11664601, -6.50709817, -3.26491035, -3.44978362,
 -13.06956505,  0.06518084,  0.58252559,-13.90539441,  8.67737978,
  -0.77470928,  0.81489966,  9.13740768, -8.52917035, 11.67090749,
  -1.59855816,-14.04290171,  4.31396527,  0.44220236, -8.4558164,
   3.37632976, 14.39894445,  2.86740211, -7.20996418,  1.42434276,
   8.79947477,  0.14433892,  4.61478049, -0.06555093,-12.86108929,
  -9.94426393,  9.49427134, -4.08600969,  3.07716718,-10.3319003,
   8.20781991,  0.16422615, -9.38236886, 12.15359352, -8.52755527,
  -3.46241861, 10.58001391, 12.42423735,  7.79527622,  2.83124337,
  -2.95177608,  7.59726873,  9.90400375,  4.54379124,  7.93149913,
  -3.43318617,-14.71053667,  0.7585574 ,-12.15699668,-14.16640498,
  11.05230297, -0.52525004, 12.14242352,  6.7639155 ,  1.59440174,
  -0.36800961, -4.79387237, -9.50851097, -3.09880187,  8.54313203,
 -13.16755967,  4.33323163, -8.97164806, -5.50526472, -3.01982127,
   1.81249791,  4.920354  , -9.42681446,-13.59758734, 12.59200511,
  10.73921559, -7.58158233, -2.44486998,  5.16125482,-14.8986748,
  -8.3822747 ,-12.20784137, 10.22237576,-13.11600716, -2.00259959,
  -7.47045216, -7.32365737,  0.59545339,-14.43135392])

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

    env = RaycastEnv(CTRNNEnv(**config.env, use_visual=False))
    
    return test(env)

    algo = EvolutionarySearch(env,pop_size=pop_size, genotype_size=genome_size, lb=lb, ub=ub, max_iter=its)
    best = algo.run()
    logger.info("Best is :")
    logger.info(best)

def test(env):
    #TestAgent(env)
    total_score = 0
    episode_num = 10000
    max_step = 500
    agent = CTRNNAgent()
    agent.set_weights(get_initial(0))
    counter = 0
    total = 10000
    labels = []
    for i in range(episode_num):
        obs = env.reset()
        agent.brain.randomize_outputs(0.5,0.51)
        for i in range(max_step):
            action = agent.Act(obs)
            counter += 1
            if counter % 5 == 0:
                #print(counter // 5)
                labels.append(obs)
                total -= 1
                if total == 0:
                    df = pd.DataFrame(labels)
                    df.to_csv("./labels.csv",index_label="index")
                    return labels
            #action = np.array([1,0])
            #print("Action:", action)
            #print("Observation", obs)

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
