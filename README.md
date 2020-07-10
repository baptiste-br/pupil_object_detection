# pupil_object_recognition
Implementing object recognition based on focus on the Pupil Core headset from Pupil Labs.

Installation : 
- follow this tutorial about the prerequisites for Yolo, open CV and using these softwares with CUDA with this tutorial from Augmented Startups : 
  https://youtu.be/5pYh1rFnNZs
- Once your installation of Yolo is complete, download weights and cfg files for the YOLO model we are using on https://github.com/AlexeyAB/darknet.
  Download yolov4-tiny.weights and yolov4-tiny.cfg for fast detection and yolov4.weights and yolov4.cfg for precise detection
  Place the .weights file in darknet\build\darknet\x64 
  and the .cfg files in darknet\build\darknet\x64\cfg.
- Then download the compiled version of Pupil Capture from https://pupil-labs.com/products/core/ and install it like a classic Windows application
- Open Pupil Capture and try it, this step is just so that it creates the setting folder.
- Go to C:\User\\"your user name"\pupil_capture_settings, then open the plugins folder.
- There you have to copy your darknet folder
- Then open the command prompt (by typing cmd in the file adress bar in the explorer), and type git clone https://github.com/baptiste-br/pupil_object_recognition.git
- You don't need git you can simply download the detection_plugin file and place it in the plugins folder next to the darknet folder.
- Open Pupil Capture again, make sure the Pupil Core are connected to the computer and then in the plugin manager you should have at the bottom the Yolov4 Detection Plugin.
