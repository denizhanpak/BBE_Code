import logging
import os
from typing import Any, List, Optional
from pl_bolts.models.autoencoders import VAE
from torchvision.models import resnet18

import numpy as np
import torch

import gym

from gym_unity.envs import UnityToGymWrapper

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityEnvironmentException, UnityWorkerInUseException
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfig,
    EngineConfigurationChannel,
)

logger = logging.getLogger(__name__)

class RaycastEnv(gym.ObservationWrapper):
    """
    Gym env wrapper that transpose visual observations to (C,H,W).
    """
    def __init__(self, env):
        super().__init__(env)

        # Assumes raycast observation space of len 20
        assert env.observation_space.shape[0] == 20
        self.observation_space = gym.spaces.Discrete(5)

    def observation(self, obs):
        rv = [0] * 5
        for i, val in enumerate(obs):
            if i % 4 == 2 or i % 4 == 3:
                continue
            if val == 1:
                if i % 4 == 0:
                    rv[i // 4] = 5
                else:
                    rv[i // 4] = -5
        
        #Rearrange raycasts so that it goes left to right
        return np.array(rv)[[4,2,0,1,3]]

class VAEEnv(gym.ObservationWrapper):
    """
    Gym env wrapper that transpose visual observations to (C,H,W).
    """
    def __init__(self, env):
        super().__init__(env)

        # Assumes raycast observation space of len 20
        self.observation_space = gym.spaces.Discrete(5)
        self.v = VAE.load_from_checkpoint("/home/denpak/logs/vae_model/last.ckpt")

        for param in self.v.parameters():
            param.requires_grad = False


    def observation(self, obs):
        obs = np.rollaxis(obs,2)
        x = torch.unsqueeze(torch.tensor(obs),0)
        x = self.v.encoder(x)
        obs = self.v.fc_mu(x)[0].detach().numpy() * 5
        return obs
                

class CNNEnv(gym.ObservationWrapper):
    """
    Gym env wrapper that transpose visual observations to (C,H,W).
    """
    def __init__(self, env):
        super().__init__(env)

        # Assumes raycast observation space of len 20
        self.observation_space = gym.spaces.Discrete(5)
        
        PATH = "/home/denpak/logs/cnn_model"
        ckpt = torch.load(PATH+"/last.ckpt")
        self.model = resnet18(num_classes=5)
        keys = ckpt['state_dict'].keys()
        state_dict = {}
        for key in keys:
            state_dict[key[6:]] = ckpt['state_dict'][key]
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False

    def observation(self, obs):
        obs = np.rollaxis(obs, 2)
        x = torch.unsqueeze(torch.tensor(obs),0)
        x = self.model(x)
        obs = x.detach().numpy()[0]
        return obs
    

class CTRNNEnv(gym.Wrapper):
    """ Gym environment wrapper for ChickAI.

    Args:
        env_path: Path to the Unity environment executable. Connects to Unity
            Editor if env_path is not provided.

        imprint_video: Path to a WEBM video file.

        test_video: Path to a WEBM video file.

        log_dir: Directory to save log from Unity environment. The directory
            will be created if it does not already exist.

        input_resolution: Size of the agent's visual inputs in pixels.

        episode_steps: Number of steps in each episode. The environment resets
            after every episode_steps.

        seed: Random seed for the Unity environment.

        test_mode: Run environment in Test Mode. In Test Mode, the agent is
            spawned from the center of the chamber at the beginning of each
            episode.

        base_port: Base port number for communicating with Unity environment.

        time_scale: Time scale of the Unity environment.

        width: The width of the executable window of the environment in pixels.

        height: The height of the executable window of the environment in pixels.
    """
    def __init__(
        self,
        env_path: Optional[str] = None,
        imprint_video: Optional[str] = None,
        test_video: Optional[str] = None,
        log_dir: Optional[str] = None,
        input_resolution: int = 64,
        episode_steps: int = 1000,
        seed: int = 0,
        test_mode: bool = False,
        base_port: int = UnityEnvironment.BASE_ENVIRONMENT_PORT,
        time_scale: int = 20,
        capture_frame_rate: int = 60,
        width: int = 80,
        height: int = 80,
        use_visual: bool = True,
        **kwargs,
    ):
        engine_config = EngineConfig(
            width=width,
            height=height,
            quality_level=5,
            time_scale=time_scale,
            target_frame_rate=-1,
            capture_frame_rate=capture_frame_rate,
        )
        env_args = _build_chickAI_env_args(
            input_resolution=input_resolution,
            episode_steps=episode_steps,
            imprint_video=imprint_video,
            test_video=test_video,
            log_dir=log_dir,
            test_mode=test_mode
        )
        agent_info_channel = FloatPropertiesChannel()
        unity_env = _make_unity_env(
            env_path=env_path,
            port=base_port,
            seed=seed,
            env_args=env_args,
            engine_config=engine_config,
            side_channels=[agent_info_channel]
        )
        env = UnityToGymWrapper(unity_env, flatten_branched=True, use_visual=use_visual)
        super().__init__(env)
        self.env = env
        self.agent_info_channel = agent_info_channel

    def step(self, action: Any):
        """ Execute one step of environment interaction.

        Args:
            action: An action to perform in the environment.

        Returns:
            next_obs: Visual observation after performing the action.
            reward: Reward value from the environment.
            done: Boolean flag indicating the end of an episode.
            info: Dict containing information about the agent's position.
        """
        next_obs, reward, done, info = self.env.step(action)
        agent_info = self.agent_info_channel.get_property_dict_copy()
        info.update(dict(agent=agent_info))
        return next_obs, reward, done, info


def _make_unity_env(
    env_path: Optional[str] = None,
    port: int = UnityEnvironment.BASE_ENVIRONMENT_PORT,
    seed: int = -1,
    env_args: Optional[List[str]] = None,
    engine_config: Optional[EngineConfig] = None,
    side_channels: Optional[List[SideChannel]]= None
) -> UnityEnvironment:
    """
    Create a UnityEnvironment.
    """
    # Use Unity Editor if env file is not provided.
    if env_path is None:
        port = UnityEnvironment.DEFAULT_EDITOR_PORT
    else:
        launch_string = UnityEnvironment.validate_environment_path(env_path)
        if launch_string is None:
            raise UnityEnvironmentException(
                f"Couldn't launch the {env_path} environment. Provided filename does not match any environments."
            )
        logger.info(f"Starting environment from {env_path}.")

    # Configure Unity Engine.
    if engine_config is None:
        engine_config = EngineConfig.default_config()

    engine_configuration_channel = EngineConfigurationChannel()
    engine_configuration_channel.set_configuration(engine_config)

    if side_channels is None:
        side_channels = [engine_configuration_channel]
    else:
        side_channels.append(engine_configuration_channel)

    # Find an available port to connect to Unity environment.
    while True:
        try:
            env = UnityEnvironment(
                file_name=env_path,
                seed=seed,
                base_port=port,
                args=env_args,
                side_channels=side_channels,
            )
        except UnityWorkerInUseException:
            logger.debug(f"port {port} in use.")
            port += 1
        else:
            logger.info(f"Connected to environment using port {port}.")
            break

    return env


def _build_chickAI_env_args(
    input_resolution: Optional[int] = None,
    episode_steps: Optional[int] = None,
    imprint_video: Optional[str] = None,
    test_video: Optional[str] = None,
    log_dir: Optional[str] = None,
    test_mode: bool = False,
) -> List[str]:
    """
    Build environment arguments that will be passed to chickAI unity environment.
    """

    # Always enable agent info side channel.
    env_args = ["--enable-agent-info-channel"]

    if input_resolution is not None:
        env_args.extend(["--input-resolution", str(input_resolution)])

    if episode_steps is not None:
        env_args.extend(["--episode-steps", str(episode_steps)])

    if imprint_video is not None:
        if not os.path.exists(imprint_video):
            raise FileNotFoundError(f"imprint_video file not found: {imprint_video}")
        video_path = os.path.abspath(os.path.expanduser(imprint_video))
        env_args.extend(["--imprint-video", video_path])

    if test_video is not None:
        if not os.path.exists(test_video):
            raise FileNotFoundError(f"test_video file not found: {test_video}")
        video_path = os.path.abspath(os.path.expanduser(test_video))
        env_args.extend(["--test-video", video_path])

    if log_dir is not None:
        env_args.extend(["--log-dir", log_dir])

    if test_mode is True:
        env_args.extend(["--test-mode"])

    return env_args
