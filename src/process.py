import numpy as np
import albumentations as alb
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import *
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import load_model, Sequential



class Image:
    def __init__(self, image, size=[224, 224]):
        self.image = image
        self.size = size
        self.aug = alb.Compose([alb.Resize(size[0], size[1])])

        
    def process(self):
        proc_img = np.array(preprocess_input(self.aug(image=self.image)['image']))
        single_sz = tuple([-1] + self.size + [3])
        processed_img = proc_img.reshape(*single_sz)

        return processed_img
        

class Model:
    
    def __init__(self, model_path):

        self.model_path = model_path
        self.model = load_model(model_path)

    def predict(self, processed_img):
        pred = self.model.predict(processed_img)
    
        return pred

        
class Ensembel:

    def __init__(self, pred1, pred2):

        self.pred1 = pred1
        self.pred2 = pred2

    def weighted_avg(self, weight1, weight2, threshold=.5):

        weighted_pred = self.pred1 * weight1 + self.pred2 * weight2
        val = 'No mask'
        if weighted_pred >= threshold:
            val = 'Mask'
        acc = '%.2f'%(weighted_pred * 100) + '%'
        return val, acc
            

        










