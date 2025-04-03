# Cheongju University Camera(OpenCV) Lecture Notes<br>
  Author : Gihoon SOng<br><br>
  Affiliation : UNICON LAB (HanYang University, South Korea)<br><br>
  Position : M.A. student<br><br>
  E-mail : sgh7344@hanyang.ac.kr<br><br>

#  Course Duration
  Date : 25.04.09

# Camera(OpenCV) Lecture Note Description
  "Camera Lecture Notes (1).pdf" file is the Camera(OpenCV) lecture notes.<br><br>
  "Camera Lecture Notes (2).pdf" file is the Camera(YOLO v8) lecture notes.<br><br>
  "Practice source code" folder is a directory that contains Python example codes related to OpenCV practice.<br><br>

# Commands for practicing OpenCV(Python) based on ROS.

<br>

**Note: This practice was conducted in an Ubuntu 20.04 LTS and ROS(Robot Operating System) 1 Noetic environment.** <br><br>

**To set up the project, follow these steps:** <br><br>

# 1. Install libraries related to 'OpenCV'
  
  <br>
  
  ```shell
  pip install opencv-python
  ```
  **or**
  ```shell
  pip3 install opencv-python
  ```
  <br>
  
  ```shell
  pip install numpy
  ```
  <br><br>

# 2. Create a package for practicing OpenCV in the ROS Noetic environment
  
  <br>
  
  ```shell
  cd ~/catkin_ws/src
  ```
  ```shell
  catkin_create_pkg opencv_test rospy std_msgs sensor_msgs cv_bridge
  ```
  <br><br>

# 3. Set up CV bridge for integrating OpenCV with the ROS environment
  
  <br>
  
  ```shell
  cd opencv_test
  ```
  <br>

  **Modify the 'CMakeLists.txt' file.**
  ```shell
  gedit CMakeLists.txt
  ```

  <br>

  **Add the following commands to the file.**
  ```shell
  include_directories(
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
  )
  ```
  <br>
  
  **Modify the 'package.xml' file.**
  ```shell
  gedit package.xml
  ```

  <br>

  **Add the following commands to the file.**
  ```shell
  <build_depend>opencv</build_depend>
  <exec_depend>opencv</exec_depend>
  ```
<br><br>

# 4. Create a 'scripts' folder for practicing OpenCV and build the catkin workspace
  
  <br>
  
  ```shell
  cd ~/catkin_ws/src/opencv_test
  ```
  ```shell
  mkdir scripts
  ```
  ```shell
  cd scripts
  ```
  <br>

  **Download the Python practice code located in the 'OpenCV example source code' folder within the 'Practice source code' folder in the GitHub repository to the 'scripts' folder.** <br><br>
  
  ```shell
  chmod +x *
  ```
  ```shell
  cd ~/catkin_ws && catkin_make
  ```
  ```shell
  source devel/setup.bash
  ```
  <br><br>

# 5. Create a 'images' folder for practicing OpenCV
  
  <br>
  
  ```shell
  cd ~/catkin_ws/src/opencv_test
  ```
  ```shell
  mkdir images
  ```
  ```shell
  cd images
  ```
  <br>

  **Download the files 'image.jpg', 'road1.jpg', 'road2.jpg' from the 'images' folder within the 'Practice source code' folder in the GitHub repository to the 'images' folder.** <br><br><br>

# 6. Perform OpenCV practice in the ROS environment
  
  <br>
  
  ```shell
  cd ~/catkin_ws/src/opencv_test/scripts
  ```
  <br>

  **Note: Ensure to set the 'image_path' variable correctly to the path on your local PC within each Python practice code.** <br><br>
  
  **6-1. Practice displaying images using OpenCV.** <br>
  ```shell
  python3 show_image.py
  ```
  <br>

  **6-2. Practice converting images to gray scale using OpenCV.** <br>
  ```shell
  python3 show_gray_scale_image.py
  ```
  <br>

  **6-3. Practice edge detection in images using OpenCV.** <br>
  ```shell
  python3 show_edge_image.py
  ```
  <br>

  **6-4. Practice contour detection in images using OpenCV.** <br>
  ```shell
  python3 show_contour_image.py
  ```
  <br>

  **6-5. Practice image rotation, resizing, and skew transformations using OpenCV.** <br>
  ```shell
  python3 show_turn_resize_affine_image.py
  ```
  <br>

  **6-6. Practice image filtering using Gaussian blur in OpenCV.** <br>
  ```shell
  python3 show_blurred_image.py
  ```
  <br>

  **6-7. Practice face detection in images using OpenCV.** <br>
  ```shell
  python3 show_face_detection_image.py
  ```
  <br>
  
  **6-8. Practice lane detection in images using Hough transform in OpenCV.** <br>
  ```shell
  python3 hough_lane_detection.py
  ```
  <br>

  **6-9. Practice lane detection in images using HSV(Hue+Saturation+Value) in OpenCV.** <br>
  ```shell
  python3 hsv_lane_detection.py
  ```
  <br>

  **6-10. Practice lane detection in images using<br> 
  Hough transform and HSV(Hue+Saturation+Value) in OpenCV.** <br>
  
  ```shell
  python3 hough_hsv_lane_detection.py
  ```
  <br><br>


# 7. Practice lane detection and tracking using the ROS Gazebo simulation environment
  
  <br>

  **7-1. ROS Gazebo simulation environment setup.** <br>
  ```shell
  cd ~/catkin_ws/src/
  ```
  <br>

  ```shell
  git clone https://github.com/gihoonbackend/ackermann_vehicle.git
  ```
  **or**
  ```shell
  git clone https://github.com/0-keun/ackermann_vehicle.git
  ```
  <br>

  ```shell
  sudo apt install ros-noetic-ackermann-msgs
  ```
  <br>

  ```shell
  cd ..
  ```
  <br>

  ```shell
  rosdep install --from-paths src --ignore-src -r –y
  ```
  <br>

  ```shell
  catkin_make
  ```
  <br>

  ```shell
  source devel/setup.bash
  ```
  <br>

  **Set Gazebo simulation environment variables in the bashrc file and perform sourcing.** <br>
  ```shell
  gedit ~/.bashrc
  ```
  <br>

  **Caution: Make sure to accurately write the path on your local PC.** <br>
  
  ```shell
  export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/home/{your PC name}/catkin_ws/src/ackermann_vehicle/roadmap_generator
  ```

  ```shell
  source ~/.bashrc
  ```
  <br>

  **7-2. Run the ROS Gazebo simulation environment.** <br>
  ```shell
  cd catkin_ws/src/ackermann_vehicle/ackermann_vehicle_gazebo/launch
  ```
  ```shell
  roslaunch ackermann_vehicle_noetic.launch
  ```
  <br>

  **If the track is not created as shown in the picture below, enter the following path and replace the image file('road_straight.png' and 'road_curve.png' file), then re-run the launch file.** <br><br>

![track_error](https://github.com/user-attachments/assets/9f928fbf-0972-41ca-8003-a9adcefab43a)<br><br><br>

  ```shell
  cd ~/catkin_ws/src/ackermann_vehicle/roadmap_generator/road_straight/materials/textures
  ```  
  **Replace the 'road_straight.png' file with the new 'road_straight.png' file located in the images folder within the Practice source code folder in the GitHub repository.** <br><br><br>
  
  ```shell
  cd ~/catkin_ws/src/ackermann_vehicle/roadmap_generator/road_curve/materials/textures
  ```
  **Replace the 'road_curve.png' file with the new 'road_curve.png' file located in the 'images' folder within the 'Practice source code' folder in the GitHub repository.** <br><br><br>
  
  **7-3. Download all the Python practice codes located in the 'Gazebo simulation source code' folder within the 'Practice source code' folder in the GitHub repository to the 'scripts' folder in the 'ackermann_vehicle_gazebo' package on your local PC.** <br><br><br>

  **7-4. Practice detecting white and yellow lanes in the Gazebo simulation track.** <br>
  ```shell
  cd catkin_ws/src/ackermann_vehicle/ackermann_vehicle_gazebo/scripts
  ```
  <br>

  ```shell
  python3 lane_detection.py
  ```
  <br>

  https://github.com/user-attachments/assets/0c233db1-7853-4785-92df-6eda489a4d02

  <br><br>

  **7-5. Practice detecting white lanes after converting the camera image to a bird's-eye view in the Gazebo simulation track.** <br>
  ```shell
  cd catkin_ws/src/ackermann_vehicle/ackermann_vehicle_gazebo/scripts
  ```
  <br>

  ```shell
  python3 bev_lane_detection.py
  ```  
  <br>

  https://github.com/user-attachments/assets/e1b4ca74-bac2-415b-a69b-c1c7d4fd4171
  
  <br><br>

  **7-6. Practice detecting and tracking white lanes after converting the camera image to a bird's-eye view in the Gazebo simulation track.** <br>
  ```shell
  cd catkin_ws/src/ackermann_vehicle/ackermann_vehicle_gazebo/scripts
  ```
  <br>

  ```shell
  python3 bev_lane_detection_and_tracking.py
  ```
  <br>

  https://github.com/user-attachments/assets/45b8b816-3115-44de-b638-36efb349b97c
  
  <br><br>

  **7-7. Practice detecting white and yellow lanes, extracting midpoints, and following them after converting the camera image to a bird's-eye view in the Gazebo simulation track.** <br>
  ```shell
  cd catkin_ws/src/ackermann_vehicle/ackermann_vehicle_gazebo/scripts
  ```
  <br>

  ```shell
  python3 bev_lane_detection_and_tracking2.py
  ```
  <br>

  https://github.com/user-attachments/assets/027ffb3b-de90-48b2-83ba-b873e99b3105

  <br><br>
