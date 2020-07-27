# Computer Pointer Controller

In this project, you will use a gaze detection model to control the mouse pointer of your computer. You will be using the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project will demonstrate your ability to run multiple models in the same machine and coordinate the flow of data between those models.
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
![alt text](https://github.com/MonicaSai7/Computer-Pointer-Controller/blob/master/bin/pipeline.png)

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
The demo of the project can be run using the following command:
```python3 src/main.py -md intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -ml intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -mh intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -mg intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i bin/demo.mp4```

## Documentation
python3 src/main.py 
--helpusage: main.py [-h] -i INPUT [-l CPU_EXTENSION] [-d DEVICE] [-pt PROB_THRESHOLD] 
            [-mg MODEL_GAZE] [-md MODEL_FACEDETECTOR] [-mh MODEL_HEADPOSE] [-ml MODEL_FACELM]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to video file or enter device cam for webcam
  -l CPU_EXTENSION, --cpu_extension CPU_EXTENSION
                        MKLDNN (CPU)-targeted custom layers.Absolute path to a
                        shared library with thekernels impl.
  -d DEVICE, --device DEVICE
                        Specify the target device to infer on: CPU, GPU, FPGA
                        or MYRIAD is acceptable. Sample will look for a
                        suitable plugin for device specified (CPU by default)
  -pt PROB_THRESHOLD, --prob_threshold PROB_THRESHOLD
                        Probability threshold for detections filtering(0.5 by
                        default)
  -mg MODEL_GAZE, --model_gaze MODEL_GAZE
                        Path to an xml file with a trained gaze detector
                        model.
  -md MODEL_FACEDETECTOR, --model_facedetector MODEL_FACEDETECTOR
                        Path to an xml file with a trained face detector
                        model.
  -mh MODEL_HEADPOSE, --model_headpose MODEL_HEADPOSE
                        Path to an xml file with a trained head pose detector
                        model.
  -ml MODEL_FACELM, --model_facelm MODEL_FACELM
                        Path to an xml file with a trained face landmarks
                        detector model.

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
