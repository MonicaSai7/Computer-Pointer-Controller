'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore

class Headpose_Estimator:
    '''
    Class for the Headpose Estimation Model.
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
        self.input_name = next(iter(self.network.inputs))
        self.output_name = next(iter(self.network.outputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_shape = self.network.outputs[self.output_name].shape
        print("IR successfully loaded into Inference Engine")

        return
    
    def predict(self, image):
        
        p_frame = self.preprocess_input(image)
        outputs = self.exec_network.infer({self.input_name: p_frame})
        return self.preprocess_output(outputs)

    def check_model(self):
        pass

    def preprocess_input(self, image):
        
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, outputs):
        result = []
        result.append(outputs['angle_y_fc'].tolist()[0][0])
        result.append(outputs['angle_p_fc'].tolist()[0][0])
        result.append(outputs['angle_r_fc'].tolist()[0][0])
        return result