from flask import Flask, request
from flask_restful import Api, Resource
import cv2
from PIL import Image
import numpy as np
import api_model.BaksaraConst as BaksaraConst
from numpy import asarray


# API purpose
        # if 'image' not in request.files:
        #     return jsonify({'error': 'No image found'}), 400
        # file = request.files['image']
        # class_input = request.form['actual_class']
        # image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        



import cv2
import numpy as np

class Kelas(Resource):
    def __init__(self):
        self.class_names = BaksaraConst.CLASS_NAMES
        self.final_model = BaksaraConst.MODEL

    def prep_predict_debug(self, image):
        image_as_array = self.PreprocessImageAsArray(image, show_output=False)
        pred = self.final_model.predict(image_as_array)
        sorted_ranks = np.flip(np.argsort(pred[0]))
        max_index = np.argmax(pred)
        return pred[max_index], self.class_names[max_index]
        
    def post(self):

        if 'image' not in request.files:
            response = {
                'error' : 'no image found'
            }
            return response
        file = request.files['image']
        class_input = request.form['actual_class']

        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("No image found")

        maxclass_prob, maxclass_name = self.prep_predict(image)
        res_debug = [maxclass_name, maxclass_prob]
        response = {
            'class': class_input,
            'prob': res_debug
        }
        return response

        try:
            predku, sorted_ranku = self.prep_predict(image)
            response_class = class_input
            response_prob = self.rules(predku, sorted_ranku, class_input)
            print(f"[KELAS][SELESAI PROSES]: {response_class} {response_prob}")
            # Rest of the code
            # index_max = 
            response = {
            'class': response_class,
            'prob': str(response_prob)
            }
            return response
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            # Handle the error condition appropriately

    def PreprocessImageAsArray(self, image, show_output=False):
        im = cv2.resize(image, (128, 128))

        image_as_array = np.expand_dims(im, axis=0)
        scaled_image_as_array = np.true_divide(image_as_array, 255)

        return scaled_image_as_array

    def take_class(self, pred, sorted_ranks, class_input):
        inputted_class_rank = 1
        rank = 1
        for class_rank in sorted_ranks:
            if self.class_names[class_rank] == class_input:
                inputted_class_rank = class_rank
            rank += 1
        return class_input, pred[0][inputted_class_rank]

    def prep_predict(self, image):
        image_as_array = self.PreprocessImageAsArray(image, show_output=False)
        pred = self.final_model.predict(image_as_array)
        sorted_ranks = np.flip(np.argsort(pred[0]))
        return pred, sorted_ranks

    def rules(self, pred, sorted_rank, class_input):
        res = []
        if class_input == 'carakan_ha':
            res1 = self.take_class(pred, sorted_rank, 'carakan_ha') or ('null', 0.0)
            res.append(res1)
            res2 = self.take_class(pred, sorted_rank, 'carakan_ta') or ('null', 0.0)
            res.append(res2)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]

        # elif class_input == 'pasangan_ra' or class_input == 'carakan_ra':
        #     res1 = self.take_class(pred, sorted_rank, 'pasangan_ra') or ('null', 0.0)
        #     res.append(res1)
        #     res2 = self.take_class(pred, sorted_rank, 'carakan_ra') or ('null', 0.0)
        #     res.append(res2)
        #     highest_tuple = max(res, key=lambda x: x[1])
        #     return highest_tuple[1]

        # elif class_input == 'pasangan_ga' or class_input == 'carakan_ga':
        #     res1 = self.take_class(pred, sorted_rank, 'pasangan_ga') or ('null', 0.0)
        #     res.append(res1)
        #     res2 = self.take_class(pred, sorted_rank, 'carakan_ga') or ('null', 0.0)
        #     res.append(res2)
        #     highest_tuple = max(res, key=lambda x: x[1])
        #     return highest_tuple[1]

        # elif class_input == 'pasangan_ya' or class_input == 'carakan_ya':
        #     res1 = self.take_class(pred, sorted_rank, 'pasangan_ya') or ('null', 0.0)
        #     res.append(res1)
        #     res2 = self.take_class(pred, sorted_rank, 'carakan_ya') or ('null', 0.0)
        #     res.append(res2)
        #     highest_tuple = max(res, key=lambda x: x[1])
        #     return highest_tuple[1]

        # elif class_input == 'pasangan_nga' or class_input == 'carakan_nga':
        #     res1 = self.take_class(pred, sorted_rank, 'pasangan_nga') or ('null', 0.0)
        #     res.append(res1)
        #     res2 = self.take_class(pred, sorted_rank, 'carakan_nga') or ('null', 0.0)
        #     res.append(res2)
        #     highest_tuple = max(res, key=lambda x: x[1])
        #     return highest_tuple[1]
        
        # elif class_input == 'sandhangan_e' or class_input == 'pasangan_wa':
        #     res1 = self.take_class(pred, sorted_rank, 'sandhangan_e') or ('null', 0.0)
        #     res.append(res1)
        #     res2 = self.take_class(pred, sorted_rank, 'pasangan_wa') or ('null', 0.0)
        #     res.append(res2)
        #     highest_tuple = max(res, key=lambda x: x[1])
        #     return highest_tuple[1]
        
        # elif class_input == 'pasangan_dha':
        #     res1 = self.take_class(pred, sorted_rank, 'pasangan_dha') or ('null', 0.0)
        #     res.append(res1)
        #     res2 = self.take_class(pred, sorted_rank, 'pasangan_tha') or ('null', 0.0)
        #     res.append(res2)
        #     highest_tuple = max(res, key=lambda x: x[1])
        #     return highest_tuple[1]
        
        # elif class_input == 'pasangan_ma' or class_input == 'sandhangan_ng':
        #     res1 = self.take_class(pred, sorted_rank, 'pasangan_ma') or ('null', 0.0)
        #     res.append(res1)
        #     res2 = self.take_class(pred, sorted_rank, 'sandhangan_ng') or ('null', 0.0)
        #     res.append(res2)
        #     highest_tuple = max(res, key=lambda x: x[1])
        #     return highest_tuple[1]
        else:
            res1 = self.take_class(pred, sorted_rank, class_input)
            res.append(res1)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]
