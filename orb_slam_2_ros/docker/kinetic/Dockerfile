FROM ros:kinetic-robot

# Set working directory
WORKDIR /home

# Get orb_slam_2_ros and the realsense package from git
RUN mkdir src
RUN cd src \
&& git clone https://github.com/appliedAI-Initiative/orb_slam_2_ros.git \
&& git clone https://github.com/IntelRealSense/realsense-ros.git


# Set up Kinetic keys
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# Set up realsense keys
RUN apt-key adv --keyserver keys.gnupg.net --recv-key C8B3A55A6F3EFCDE
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE

# Update
RUN apt update
RUN apt-get install software-properties-common apt-utils -y

#Add realsense repo
RUN add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u

# Install required realsense and ROS packages
RUN apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev \
    librealsense2-dbg ros-kinetic-rgbd-launch ros-kinetic-tf2-geometry-msgs python-catkin-tools -y

# Install ROS dependencies
RUN rosdep update \
&& rosdep install --from-paths src --ignore-src -r -y --skip-keys=librealsense2

# build ros package source
RUN catkin config \
      --extend /opt/ros/$ROS_DISTRO && \
    catkin build

RUN echo "source /home/devel/setup.bash" >> /$HOME/.bashrc

CMD "bash"