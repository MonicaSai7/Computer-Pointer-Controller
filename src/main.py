import os
import numpy as np
import cv2
import time
import logging
from argparse import ArgumentParser
from input_feeder import InputFeeder
from face_detection import Face_Detection
from gaze_estimation import Gaze_Estimation
from facial_landmarks_detection import Face_LandMarker
from head_pose_estimation import Headpose_Estimator
from mouse_controller import MouseController

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to video file or enter device cam for webcam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    
    parser.add_argument("-mg", "--model_gaze", required=False, type=str, default=None,
                        help="Path to an xml file with a trained gaze detector model.")
    parser.add_argument("-md", "--model_facedetector", required=False, type=str, default=None,
                        help="Path to an xml file with a trained face detector model.")
    parser.add_argument("-mh", "--model_headpose", required=False, type=str, default=None,
                        help="Path to an xml file with a trained head pose detector model.")
    parser.add_argument("-ml", "--model_facelm", required=False, type=str, default=None,
                        help="Path to an xml file with a trained face landmarks detector model.")

    return parser

def get_facedetector(args):
    model_facedetector = args.model_facedetector
    if model_facedetector:
        face_detector = Face_Detection(model_name=model_facedetector, device=args.device, extensions=args.cpu_extension)
    return face_detector

def get_gazestimator(args):
    model_gaze = args.model_gaze
    if model_gaze:
        gaze_estimator = Gaze_Estimation(model_name=model_gaze, device=args.device, extensions=args.cpu_extension)
    return gaze_estimator

def get_facelandmarker(args):
    model_facelm = args.model_facelm
    if model_facelm:
        face_landmarker = Face_LandMarker(model_name=model_facelm, device=args.device, extensions=args.cpu_extension)
    return face_landmarker

def get_headpose(args):
    model_headpose = args.model_headpose
    if model_headpose:
        headpose_est = Headpose_Estimator(model_name=model_headpose, device=args.device, extensions=args.cpu_extension)
    return headpose_est

def main():
    # Grab command line args
    args = build_argparser().parse_args()
    logger = logging.getLogger()
    
    ## Handle input
    if args.input == 'CAM':
        input_feed = InputFeeder('cam')
    else:
        try:
            input_feed = InputFeeder('video', args.input)
        except:
            print("Unable to access input video file.")
            exit()
    
    start_load = time.time()

    # Get models
    facedetector = get_facedetector(args)
    facelm = get_facelandmarker(args)
    headpose = get_headpose(args)
    gaze = get_gazestimator(args)

    mouse_controller = MouseController(precision='medium', speed='fast')

    input_feed.load_data()
    facedetector.load_model()
    facelm.load_model()
    headpose.load_model()
    gaze.load_model()

    total_load_time = time.time() - start_load

    n_frames = 0
    count = 0
    inference_time = 0
    for frame in input_feed.next_batch():
        if frame is not None:
            n_frames += 1
            cv2.imshow('video', cv2.resize(frame, (500, 500)))
            
            key = cv2.waitKey(60)
            start_inference = time.time()

            ### Face Detection
            face, coords = facedetector.predict(frame, args.prob_threshold)
            if type(face) == int:
                logger.error("No face detected")
                continue
            fd_time = round(time.time() - start_inference, 5)
            #logger.info("Face detection took {} seconds.".format(fd_time))

            ### Headpose Estimation
            headpose_out = headpose.predict(face)

            ### Face Landmarks Extraction
            left_eye, right_eye, eye_coords = facelm.predict(face)

            ### Gaze Estimation
            mouse_coord, gaze_vec = gaze.predict(left_eye, right_eye, headpose_out)

            stop_inference = time.time()
            inference_time += stop_inference - start_inference
            count += 1

            img = cv2.resize(frame, (500, 500))
            cv2.imshow("Visualize", img)
            mouse_controller.move(mouse_coord[0], mouse_coord[1])

            if key == 27:
                break
    
    fps = n_frames/inference_time
    logger.error("Total loading time: " + str(total_load_time) + 's')
    logger.error("Total inference time {}s".format(inference_time))
    logger.error("Frames per second: {} fps".format(fps))

    cv2.destroyAllWindows()
    input_feed.close()

if __name__ == '__main__':
    main()

