# --- Base Image ---
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- Install System Dependencies ---
# Includes Java (default-jre) for the MATLAB installer
RUN apt-get update && apt-get install -y \
    wget unzip xorg default-jre libncurses6 \
    libxext6 libx11-6 libglu1-mesa libxi6 libxtst6 libxrender1 \
    libxrandr2 libxinerama1 libxcursor1 libxcomposite1 libxdamage1 \
    libfontconfig1 libxss1 libasound2 \
    python3.10 python3.10-distutils python3-pip \
    python3-tk \
    python3-pyqt5 \
    libqt5gui5 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    xorg \
    default-jre \
    libncurses6 \
    libxext6 \
    libx11-6 \
    libglu1-mesa \
    libxtst6 \
    libxmu6 \
    libxi6

# --- Install Python Dependencies ---
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# --- Setup MATLAB Runtime Environment Variables ---
ENV MCR_ROOT=/opt/mcr/R2025a
ENV LD_LIBRARY_PATH=${MCR_ROOT}/runtime/glnxa64:${MCR_ROOT}/bin/glnxa64:${MCR_ROOT}/sys/os/glnxa64:${LD_LIBRARY_PATH}
ENV XAPPLRESDIR=${MCR_ROOT}/X11/app-defaults

# --- Install MATLAB Runtime ---
WORKDIR /opt/mcr

# MODIFIED RUN command for MATLAB Runtime installation
# MODIFIED RUN command for MATLAB Runtime installation
RUN wget --progress=bar:force https://ssd.mathworks.com/supportfiles/downloads/R2025a/Release/0/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2025a_glnxa64.zip \
    && unzip -q MATLAB_Runtime_R2025a_glnxa64.zip -d mcr_install \
    && rm MATLAB_Runtime_R2025a_glnxa64.zip \
    && chmod +x mcr_install/install \
    && ./mcr_install/install -mode silent -agreeToLicense yes -destinationFolder /opt/mcr -logfile /tmp/mcr_log.txt \
    && if [ ! -f /opt/mcr/R2025a/bin/glnxa64/libmwlaunchermain.so ]; then \
        echo "FATAL: MATLAB Runtime installation failed. Key library not found." >&2; \
        echo "--- Installer Log ---" >&2; \
        cat /tmp/mcr_log.txt >&2; \
        exit 1; \
       fi \
    && rm -rf mcr_install

# --- Copy Application Files ---
WORKDIR /app
COPY runSimulationWithAnimation run_runSimulationWithAnimation.sh /app/
COPY main_run_scenarios run_main_run_scenarios.sh /app/
COPY scenarios/ /app/scenarios/
COPY energy_agent/ /app/energy_agent/
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

# --- Set Permissions and Environment ---
RUN chmod +x runSimulationWithAnimation run_runSimulationWithAnimation.sh main_run_scenarios run_main_run_scenarios.sh /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
ENV DISPLAY=:0.0
CMD ["/bin/bash"]