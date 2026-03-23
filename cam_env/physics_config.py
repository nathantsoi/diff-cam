"""Shared physics constants for both single and batched CNC simulators.

Both `simulator.py` and `batched_simulator.py` should import from here
to keep the physics in sync. Change values here to affect both.
"""


# --- Stock Geometry ---
STOCK_CENTER = (0.5, 0.5, 0.5)
STOCK_HALF_SIZE = 0.4  # Half-extent of the stock cube

# --- Target Geometry ---
TARGET_CENTER = (0.5, 0.5, 0.5)
TARGET_RADIUS = 0.25  # Sphere target

# --- Tool Geometry ---
TOOL_START_POS = (0.0, 0.5, 0.5)  # Starting position (left edge, centered Y/Z)
TOOL_RADIUS = 0.1
TOOL_HEIGHT = 0.3

# --- Reward Function ---
TIME_PENALTY = -0.02
COMPLETION_BONUS = 10.0
COMPLETION_THRESHOLD = 1e-3
PROXIMITY_COEF_INITIAL = 0.1
PROXIMITY_ANNEAL_FRACTION = 0.5  # Anneal over this fraction of total_timesteps
PROXIMITY_CLIP = 0.009
BLOCKED_PENALTY = -0.05
