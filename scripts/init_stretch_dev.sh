#!/bin/bash
# Description: Run the docker container with GPU support

# Make sure it fails if we see any errors
set -e

# Function to check if user is in Docker group
is_in_docker_group() {
    groups | grep -q docker
}

# Function to run Docker command
run_docker_command() {
    if is_in_docker_group; then
        echo "User is in Docker group. Running command without sudo."
        docker "$@"
    else
        echo "User is not in Docker group. Running command with sudo."
        echo "To run without sudo, add your user to the docker group: sudo usermod -aG docker $USER"
        echo "Then log out and log back in."
        echo "Alternately, you can change for the current shell with newgrp: newgrp docker"
        sudo docker "$@"
    fi
}


echo "Starting Stretch AI Development Environment on $HELLO_FLEET_ID"
echo "========================================================="
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"


VERSION="0.0.1"
echo "Docker image version: $VERSION"

# sudo chown -R $USER:$USER /home/$USER/stretch_user
# sudo chown -R $USER:$USER /home/$USER/ament_ws/install/stretch_description/share/stretch_description/urdf

echo "Running docker image hellorobotinc/stretch-ai-ros2-dev:$VERSION"
# Make sure the image is up to date

docker run -it --rm \
    --net=host \
    --privileged=true \
    --device /dev/snd \
    --group-add=audio \
    -e DISPLAY=$DISPLAY \
    -e HELLO_FLEET_ID=$HELLO_FLEET_ID \
    -e XAUTHORITY=/tmp/.docker.xauth \
    -v $XAUTHORITY:/tmp/.docker.xauth:rw \
    -v /dev:/dev \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /run/dbus/:/run/dbus/:rw \
    -v /dev/shm:/dev/shm \
    -v /home/$USER/stretch_user:/home/hello-robot/stretch_user_copy \
    -v /home/$USER/ament_ws/:/home/hello-robot/ament_ws/ \
    -v /home/$USER/stretch_ai/:/home/hello-robot/stretch_ai/ \
    anranzzz/stretch-ai_ros2-dev:$VERSION \
    bash -c "source /home/hello-robot/.bashrc; \
            cd /home/hello-robot/ament_ws/src;  \
            ln -s /home/hello-robot/stretch_ai/src/stretch_ros2_bridge ~/ament_ws/src/stretch_ros2_bridge; \
            cd /home/hello-robot/ament_ws; \
            colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --symlink-install --event-handlers console_direct+; \
            cp -rf /home/hello-robot/stretch_user_copy/* /home/hello-robot/stretch_user; \
            export HELLO_FLEET_ID=$HELLO_FLEET_ID; \
            cd /home/hello-robot/ament_ws; \
            source /home/hello-robot/ament_ws/install/setup.bash; \
            ros2 launch stretch_ros2_bridge server.launch.py"
