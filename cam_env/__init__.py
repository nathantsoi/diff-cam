from gymnasium.envs.registration import register

register(
    id="CamEnvDiff-v0",
    entry_point="cam_env.cam_env:CamEnvDiff",
    kwargs={'resolution': 64, 'max_steps': 1000}
)

register(
    id="CamEnvDisc-v0",
    entry_point="cam_env.cam_env_voxel:CamEnvDisc",
    kwargs={'resolution': 64, 'max_steps': 100}
)

