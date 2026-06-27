# TACC Installation & Setup

This repository includes automated scripts to handle environment setup on TACC (Texas Advanced Computing Center) systems.

> Always clone and install this repository in your **`$SCRATCH`** directory.
> TACC `$HOME` directories have a strict 10GB limit. Python environments and dataset caches will fill this immediately, causing system errors.

## 1. First-Time Setup

1. ssh into TACC:
```bash
ssh -m hmac-sha2-512 neilvakharia45@ls6.tacc.utexas.edu
```
Password is your TACC password and Passcode is your TACC Token from Duo.


2. Navigate to Scratch and clone the repo:
```bash
cd $SCRATCH
git clone https://github.com/nathantsoi/diff-cam.git
cd diff-cam
```


3. Make the setup scripts executable:
```bash
chmod +x scripts/*.sh
```


4. **Initialize the Machine:**
Run this script once to install `uv` (if missing) and configure cache directories to use `$SCRATCH` instead of `$HOME`.
```bash
./scripts/00_init_machine.sh
```


---

## 2. Installation Methods

Choose **one** of the following methods to install dependencies.

### Option A: Compute Node + Uv

Uses uv sync which requires a dev node

1. **Request an interactive Dev Node (A100):**
```bash
idev -p gpu-a100-dev -N 1 -n 1 -t 01:00:00

```

*Wait until the command finishes and your prompt changes (indicating you are on a compute node).*
2. **Run the installer:**
```bash
# Ensure you are in the repo folder
cd $SCRATCH/diff-cam

# Run the script
./scripts/01_install_compute.sh
```

### Option B: Login Node Only via Pip 

Use this method if you cannot get a compute node or prefer standard `pip`.

1. **Run the installer directly on the login node:**
```bash
./scripts/01_install_login.sh

```


---

## 3. Usage

To run the project later, simply activate the environment:

```bash
source .venv/bin/activate

```
