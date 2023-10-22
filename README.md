# RCCAR_YOLO_BYTETRACK
Creating an rc car which follows certain object we want. (using raspberry-pi RC CAR)
  - RC CAR contains raspberry-pi so it can communicate with my laptop using python.
  - first, RC CAR sends its front realtime video to laptop.
  - second, using YOLOv8x(object detection) and BYTETRACK(object tracking), laptop determines the actuator values that can follow the certain target. (target is determined using tracking id)
  - third, laptop sends actuator values to RC CAR. then we're done.
  - In order to make this systems, we have to handle multiple skills like showing raspberry-pi's shell in my laptop using putty. etc..
  - that multiple skills is demonstrated very very detaily in my report(first file).[Creating a RC CAR which follows certain object we 4918150c91044622b6acbf56dc5c3a17.pdf]
  - laptop code : laptop.py
  - raspberry-pi code : rasp.py

in this project, we can learn about object detection, tracking, how to use linux... etc.
my report will be very very helpful for you.

If you have any questions, plz feel free to contact me. 
Email : jhss4475@dgist.ac.kr
