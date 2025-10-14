#!/bin/bash
# entrypoint.sh

# Set the MATLAB Runtime environment variables
export LD_LIBRARY_PATH=/opt/mcr/R2025a/runtime/glnxa64:/opt/mcr/R2025a/bin/glnxa64:/opt/mcr/R2025a/sys/os/glnxa64:/opt/mcr/R2025a/sys/opengl/lib/glnxa64

# Execute the command passed to the container (e.g., ./run_script.sh ...)
exec "$@"