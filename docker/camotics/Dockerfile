FROM debian:stretch
RUN printf 'deb http://archive.debian.org/debian stretch main\n' > /etc/apt/sources.list && \
    apt-get -o Acquire::Check-Valid-Until=false update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libv8-3.14.5 libglu1-mesa libqt5core5a libqt5gui5 libqt5opengl5 debconf libssl1.1 libqt5websockets5 && \
    rm -rf /var/lib/apt/lists/*
COPY camotics_1.2.0_amd64.deb /tmp/
RUN dpkg -i /tmp/camotics_1.2.0_amd64.deb && rm /tmp/camotics_1.2.0_amd64.deb
WORKDIR /work
