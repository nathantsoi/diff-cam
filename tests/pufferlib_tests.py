import gymnasium as gym
import pufferlib
import pufferlib.vector

from cam_env.cam_env import CamEnv

def make_env():
    def thunk():
        env = gym.make("CamEnv-v0", resolution=64, max_steps=100)
        return env
    return thunk

print("Got here")

# Test 1: What does the thunk return?
env = make_env()()
print(f"Raw env type: {type(env)}")
print(f"Has observation_space: {hasattr(env, 'observation_space')}")
print(f"Has single_observation_space: {hasattr(env, 'single_observation_space')}")

# Test 2: What does pufferlib.vector.make return?
envs = pufferlib.vector.make(
    make_env(),
    num_envs=2,
    backend=pufferlib.vector.Serial,
)

print(f"\nVectorized env type: {type(envs)}")
print(f"Vectorized env __class__.__mro__: {type(envs).__mro__}")
print(f"Has observation_space: {hasattr(envs, 'observation_space')}")
print(f"Has single_observation_space: {hasattr(envs, 'single_observation_space')}")

# Test 3: Check what attributes the vectorized env has
print(f"\nAll public attributes:")
for attr in dir(envs):
    if not attr.startswith('_'):
        print(f"  {attr}")