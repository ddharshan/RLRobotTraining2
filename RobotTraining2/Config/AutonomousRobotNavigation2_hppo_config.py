from easydict import EasyDict

#change - 1st two lines------------------------------------------------------

AutonomousRobotNavigation2_hppo_config = dict(    
    exp_name='AutonomousRobotNavigation2_hppo_seed0',
    env=dict(
        collector_env_num=8, #default is 8
        evaluator_env_num=5,
        # (bool) Scale output action into legal range, usually [-1, 1].
        act_scale=True,
        env_id='Moving-v0',  # ['Sliding-v0', 'Moving-v0']
        n_evaluator_episode=5,
        stop_value=1,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        action_space='hybrid',
        recompute_adv=True,
        model=dict(
            obs_shape=20,   #OBS--------------------------------------------------------------------
            action_shape=dict(
                action_type_shape=3,
                action_args_shape=2,
            ),
            action_space='hybrid',
            encoder_hidden_size_list= [256,128, 64, 64],
            sigma_type='fixed',
            fixed_sigma_value=0.3,
            bound_type='tanh',
        ),
        learn=dict(
            epoch_per_collect=10,  #default is 10
            batch_size=320,
            learning_rate=5e-4, #The default is 3e-4
            value_weight=0.5,
            entropy_weight=0.03,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_sample=int(3200),
            discount_factor=0.99,
            gae_lambda=0.95,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=200, ), ),
    ),
)

AutonomousRobotNavigation2_hppo_config = EasyDict(AutonomousRobotNavigation2_hppo_config)  #change------------------------------------
main_config = AutonomousRobotNavigation2_hppo_config    #change--------------------------------------------------------------


#change--3lines below bracket---------------------------------------------------

AutonomousRobotNavigation2_hppo_create_config = dict(
    env=dict(
        type='AutonomousRobotNavigation2',
        import_names=['RobotTraining2.Envs.AutonomousRobotNavigation2_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)

AutonomousRobotNavigation2_hppo_create_config = EasyDict(AutonomousRobotNavigation2_hppo_create_config)   #change----------------------------------
create_config = AutonomousRobotNavigation2_hppo_create_config  #change---------------------------------------------------------

if __name__ == "__main__":
    
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
