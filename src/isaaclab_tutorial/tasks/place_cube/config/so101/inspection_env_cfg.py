"""Inspect the robot and the two-cube scene before any task is defined."""

from isaaclab.envs import ManagerBasedRLEnvCfg, VideoRecorderCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.utils import preset

from .env_cfg import PhysicsCfg, SO101CubeSceneCfg


@configclass
class EmptyActionsCfg:
    pass


@configclass
class JointObservationsCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class InspectionObservationsCfg:
    policy: JointObservationsCfg = JointObservationsCfg()


@configclass
class InspectionEventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})


@configclass
class SO101CubeInspectionEnvCfg(ManagerBasedRLEnvCfg):
    scene: SO101CubeSceneCfg = SO101CubeSceneCfg(num_envs=1, env_spacing=0.9, replicate_physics=True)
    actions: EmptyActionsCfg = EmptyActionsCfg()
    observations: InspectionObservationsCfg = InspectionObservationsCfg()
    events: InspectionEventsCfg = InspectionEventsCfg()
    rewards = None
    terminations = None

    def __post_init__(self):
        self.episode_length_s = 20.0
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.64, 0.0, 0.36), lookat=(0.19, 0.02, 0.075))
        self.video_recorders = preset(
            default=[],
            record_video=[
                VideoRecorderCfg(
                    source="visualizer:newton",
                    output_dir="videos/inspection_cube",
                    video_length=120,
                    fps=30,
                )
            ],
        )
