import os
import gym
import torch
from tensorboardX import SummaryWriter
from easydict import EasyDict
from functools import partial

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.envs import get_vec_env_setting, create_env_manager
from ding.policy import  PPOPolicy
from ding.model import VAC   
from ding.utils import set_pkg_seed
from ding.rl_utils import get_epsilon_greedy_fn
from RobotTraining2.Config.AutonomousRobotNavigation2_hppo_config import AutonomousRobotNavigation2_hppo_config, AutonomousRobotNavigation2_hppo_create_config  #change------------------------------


def main(main_cfg, create_cfg, seed=0):
    cfg = compile_config(
        main_cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        AdvancedReplayBuffer,
        create_cfg=create_cfg,
        save_cfg=True
    )

    create_cfg.policy.type = create_cfg.policy.type + '_command'
    env_fn = None
    cfg = compile_config(cfg, seed=seed, env=env_fn, auto=True, create_cfg=create_cfg, save_cfg=True)
    # Create main components: env, policy
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])

    evaluator_env.enable_save_replay('AutonomousRobotNavigation2_hppo_seed0/video')  # switch save replay interface change-----------------------------------------

    # Set random seed for all package and instance
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # Set up RL Policy
    model = VAC(**cfg.policy.model) 
    policy = PPOPolicy(cfg.policy, model=model)
    policy.eval_mode.load_state_dict(torch.load("./AutonomousRobotNavigation2_hppo_seed0/ckpt/ckpt_best.pth.tar", map_location='cpu')) #change----------------------------------------------------

    
    # evaluate
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator.eval()


if __name__ == "__main__":
    # gym_hybrid environmrnt rendering is using API from "gym.envs.classic_control.rendering"
    # which is abandoned in gym >= 0.22.0, please check the gym version before rendering.
    main(AutonomousRobotNavigation2_hppo_config, AutonomousRobotNavigation2_hppo_create_config, seed=0) #change----------------------------------------------------
    
