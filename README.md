# Computer Pointer Controller

In this project, you will use a gaze detection model to control the mouse pointer of your computer. You will be using the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project will demonstrate your ability to run multiple models in the same machine and coordinate the flow of data between those models.</br></br>
### How it works
You will be using the InferenceEngine API from Intel's OpenVino ToolKit to build the project. The gaze estimation model requires three inputs:
- The head pose
- The left eye image
- The right eye image</br>

To get these inputs, you will have to use three other OpenVino models:
- Face Detection
- Head Pose Estimation
- Facial Landmarks Detection
### The Pipeline
You will have to coordinate the flow of data from the input, and then amongst the different models and finally to the mouse controller. The flow of data will look like this:

## Project Set Up and Installation
```
Computer-Pointer-Controller
│   README.md
│   requirements.txt   
│
└───bin
│   │   demo.mp4
│   
└───intel
│   └───face-detection-adas-binary-0001
│   └───gaze-estimation-adas-0002
│   └───head-pose-estimation-adas-0001
|   └───landmarks-regression-retail-0009
|
└───src
    │   input_feeder.py
    │   model.py
    |   mouse_controller.py
```
- Initialize OpenVINO environment</br>
```source /opt/intel/openvino/bin/setupvars.sh```</br></br>
- Download required models</br>
  1. face-detection-adas-binary-0001</br>
```./downloader.py --name face-detection-adas-binary-0001 --output_dir ~/IntelEdgeAI/Computer-Pointer-Controller```
  2. gaze-estimation-adas-0002</br>
  ```./downloader.py --name gaze-estimation-adas-0002 --output_dir ~/IntelEdgeAI/Computer-Pointer-Controller```
  3. head-pose-estimation-adas-0001</br>
  ```./downloader.py --name head-pose-estimation-adas-0001 --output_dir ~/IntelEdgeAI/Computer-Pointer-Controller```
  4. landmarks-regression-retail-0009</br>
  ```./downloader.py --name landmarks-regression-retail-0009 --output_dir ~/IntelEdgeAI/Computer-Pointer-Controller```</br></br>
- Install dependencies</br>
```pip install -r requirements.txt```
  
## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
