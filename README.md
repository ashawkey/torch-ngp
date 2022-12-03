# Repo containing all code required for AA275 Project (NeRF + ORB SLAM).

# How to get working

Place this repo in a `catkin_ws/src/`.

Download data to `data/village_kiiti/`.

Edit the first line of `configs/Realsense_dataset_straight_road.config`
with the proper path to your downloaded data.

Do a catkin build in `src/`.

Run orb slam with roslaunch `orb_slam2_ros orb_slam2_r200_stereo.launch`

Run data publisher with `rosrun data_publisher data_publisher.py <config file>`
For example, you could run:
`rosrun data_publisher data_publisher.py /home/chris/Project/catkin_ws/src/AA275_NeRF_SLAM_Project/configs/Realsense_dataset_straight_road.config`.

Then run the subscriber //TODO

