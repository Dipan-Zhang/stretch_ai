## build the demo docker image
`docker build -t  anranzzz/stretch-ai_ros2-dev:0.0.1  . -f Dockerfile.ros2-dev`

## create a soft link from stretch_ai to ament/src (if you haven't done it)
```
cd /home/ament_ws/src
ln -s /home/hello-robot/stretch_ai/src/stretch_ros2_bridge stretch_ros2_bridge
```
## run the docker container for robot control
`./scripts/init_stretch_dev.sh`


## Send motion using from Egoprior
`python ./policy_server_ros2/5_pickup_bottle.py`
