from gymnasium.envs.registration import register

register(
    id="CamEnv-v0",
    entry_point="cam_env.cam_env:CamEnv",
    kwargs={'resolution': 64, 'max_steps': 1000}
)

register(
    id="CamEnvVoxel-v0",
    entry_point="cam_env.cam_env_voxel:CamEnv",
    kwargs={'resolution': 64, 'max_steps': 100}
)

