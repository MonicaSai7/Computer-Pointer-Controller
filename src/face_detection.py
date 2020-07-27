'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IECore

class Face_Detection:
    '''
    Class for the Face Detection Model.
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
    
    def predict(self, image, prob_threshold):
        
        self.threshold = prob_threshold
        p_frame = self.preprocess_input(image)
        outputs = self.exec_network.infer({self.input_name: p_frame})
        coords = self.preprocess_output(outputs[self.output_name])
        face_cropped = self.crop_face(coords, image)

        if type(face_cropped) == int:
            return 0, 0

        return face_cropped, coords

    def crop_face(self, coords, image):
        if len(coords) == 0:
            return 0
        
        coords = coords[0]
        self.h = image.shape[0]
        self.w = image.shape[1]

        coords = coords* np.array([self.w, self.h, self.w, self.h])
        coords = coords.astype(np.int32)

        return image[coords[1]:coords[3], coords[0]:coords[2]]

    def check_model(self):
        pass

    def preprocess_input(self, image):
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

    def preprocess_output(self, outputs):
        coordinates = []
        for obj in outputs[0][0]:
            ### Draw bounding box if exceeding threshold
            
            if obj[2] > self.threshold:
                x_min = obj[3]
                y_min = obj[4]
                x_max = obj[5]
                y_max = obj[6]
                coordinates.append([x_min, y_min, x_max, y_max])
        return coordinates
