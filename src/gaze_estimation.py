'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import math
import numpy as np
from openvino.inference_engine import IECore

class Gaze_Estimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.plugin = None
        self.network = None
        self.input_name = None
        self.output_name = None
        self.exec_network = None
        self.infer_request = None
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_bin = self.model_name.split('.')[0] + '.bin'
        self.model_xml = model_name

    def load_model(self):
        self.plugin = IECore()
        self.network = self.plugin.read_network(model=self.model_xml, weights=self.model_bin)

        ### Check for supported layers and add any necessary extensions
        if self.extensions and 'CPU' in self.device:
            self.plugin.add_extension(self.extensions, self.device)
        
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
        
        self.exec_network = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_name = [i for i in self.network.outputs.keys()]
        print("IR successfully loaded into Inference Engine")

        return
    
    def predict(self, left_eye, right_eye, headpose_angles):

        p_left_eye, p_right_eye = self.preprocess_input(left_eye), self.preprocess_input(right_eye)
        outputs = self.exec_network.infer({'head_pose_angles': headpose_angles, 'left_eye_image': p_left_eye, 'right_eye_image': p_right_eye})
        coords, gaze_vec = self.preprocess_output(outputs, headpose_angles)
        return coords, gaze_vec

    def check_model(self):
        pass

    def preprocess_input(self, image):
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, outputs, headpose_angles):
        gaze_v = outputs[self.output_name[0]].tolist()[0]
        angle = headpose_angles[2]
        
        sine = math.sin(angle*math.pi/180.0)
        cos = math.cos(angle*math.pi/180.0)

        x = gaze_v[0] * cos + gaze_v[1] * sine
        y = gaze_v[0] * sine + gaze_v[1] * cos

        return (x, y), gaze_v
