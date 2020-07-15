# Plugin for Pupil Capture : Focused object detection
Implementing object recognition based on where you are looking at with the Pupil Core headset from Pupil Labs.

Installation : 
- follow this tutorial from Augmented Strart-ups about the prerequisites for Open CV, Yolo and using these softwares with CUDA : 
  https://youtu.be/5pYh1rFnNZs. You need to watch the 2 first episode of the series.
- Once your installation of Yolo is complete, download weights and cfg files for the YOLO model we are using on https://github.com/AlexeyAB/darknet.
  Download yolov4-tiny.weights and yolov4-tiny.cfg for fast detection and yolov4.weights and yolov4.cfg for precise detection
  Place the .weights file in darknet\build\darknet\x64 
  and the .cfg files in darknet\build\darknet\x64\cfg.
- Then download the compiled version of Pupil Capture from https://pupil-labs.com/products/core/ and install it like a classic Windows application
- Open Pupil Capture and try it, this step is just so that it creates the setting folder.
- Go to C:\User\\"your user name"\pupil_capture_settings, then open the plugins folder and copy your darknet folder there.
- Then open the command prompt (by typing cmd in the file adress bar in the explorer), and type git clone https://github.com/baptiste-br/pupil_object_recognition.git
- You don't need git you can simply download the detection_plugin file and place it in the plugins folder next to the darknet folder.
- Open Pupil Capture again, make sure the Pupil Core are connected to the computer and then in the plugin manager you should have at the bottom the Yolov4 Detection Plugin.

Description : 
- You can choose between 2 options, "speed" and "precision" : 
  -Speed option gives you a really fluid analysis of the environnement but is not very accurate.
  -Precision gives you a much more confident detection, but the video will be reduced to around 15 fps due to longer analysis.
- You have the "Load model" button. When you click it, hte model used for detection will be loaded. Note that when you are using Precision setting, it takes 10 to 15 seconds to load.
- You can check the option "open in new window" if you want to open a new window a run the object detection in it.
- You can check "only detect focused object" to only have one object to be detected and only if you are looking at it. The object you are looking at will be in a red box
- You need to check "Activate object detection" if you want the start detecting object on the video stream. When detection is activated, you will have blue boxes around objects that are detected, with at the top of the box the name of the object and the confidence score associated. Note that if you check "only detect focused object" you will not see these blue boxes, only the red one.
