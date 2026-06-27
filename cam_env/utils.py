import os
import sys

def load_env_or_abort():
    """Load environment variables from .env file or abort with instructions if missing."""
    env_path = ".env"
    if not os.path.exists(env_path):
        # Fallback to the project root relative to this file's directory (cam_env/)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        root_env_path = os.path.join(os.path.dirname(this_dir), ".env")
        if os.path.exists(root_env_path):
            env_path = root_env_path
        else:
            print("ERROR: .env file is missing!", file=sys.stderr)
            print("Please create it by copying .env.example:", file=sys.stderr)
            print("  cp .env.example .env", file=sys.stderr)
            sys.exit(1)

    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                parts = line.strip().split("=", 1)
                if len(parts) == 2:
                    os.environ[parts[0].strip()] = parts[1].strip()
