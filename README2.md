
## Common Commands

- init conda env
`conda activate stretch_ai`

- data collection:
`python3 -m stretch.app.dex_teleop.ros2_leader --task-name make_coffee --teleop-mode base_x --clutch`

- run a trained policy
`python3 -m stretch.app.lfd.ros2_lfd_leader_ee --robot_ip 192.168.1.5 --policy_path /home/chenh/anran_ws/lerobot/outputs/train/2025-12-12/19-06-43_stretch_real_cartesian_stretch_diffusion_g_abs/checkpoints/100000/pretrained_model  --policy_name diffusion --teleop-mode base_x`



## Vidbot
`conda activate vidbot`
`python policy_server_ros2/pickup_thing.py`