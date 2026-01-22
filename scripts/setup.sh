#!/bin/bash

set -x
set -e

# Install Xfce4 and TurboVNC
sudo apt-get update && sudo apt-get install -y \
    xfce4 \
    xfce4-goodies \
    dbus-x11

if [ -z "$TARGETARCH" ]; then
    TARGETARCH=$(dpkg --print-architecture)
fi

if [ "$TARGETARCH" = "amd64" ]; then \

    wget https://github.com/TurboVNC/turbovnc/releases/download/3.2.1/turbovnc_3.2.1_amd64.deb -O /tmp/turbovnc.deb; \
elif [ "$TARGETARCH" = "arm64" ]; then \
    wget https://github.com/TurboVNC/turbovnc/releases/download/3.2.1/turbovnc_3.2.1_arm64.deb -O /tmp/turbovnc.deb; \
fi && \
sudo dpkg -i /tmp/turbovnc.deb && \
rm /tmp/turbovnc.deb

# If UV_ENV_FILE=.env is not set, warn the user to set it in their environmental config
if [ -z "$UV_ENV_FILE" ]; then
    echo "Please set UV_ENV_FILE to .env by default in your environmental config"
    echo "  For example, add the following to your .zshrc or .bashrc:"
    echo "    export UV_ENV_FILE=.env"
fi
