"""RSL-RL PPO configuration for the cube-stacking state teacher."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


# BEGIN teacher-ppo
@configclass
class SO101CubeStatePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """State teacher: fully observed actor, privileged critic. Same algorithm and network as the vial teacher."""

    seed = 42
    num_steps_per_env = 64
    # The vial teacher converged in 800 iterations; stacking adds a placement stage, so allow more.
    max_iterations = 1500
    save_interval = 50
    experiment_name = "so101_cube_state"
    run_name = ""
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        # Clamp the std: with clipped actions and an entropy bonus, PPO otherwise grows it without bound.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.2, std_type="log", std_range=(0.05, 0.3)),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# END teacher-ppo
