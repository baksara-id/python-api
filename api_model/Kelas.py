from flask import Flask, request
from flask_restful import Api, Resource
import cv2
from PIL import Image
import numpy as np
import api_model.BaksaraConst as BaksaraConst
# import BaksaraConst as BaksaraConst
from numpy import asarray


class Kelas(Resource):
    def __init__(self):
        self.class_names = BaksaraConst.CLASS_NAMES
        self.final_model = BaksaraConst.MODEL

    def prep_predict_debug(self, image):
        image_as_array = self.PreprocessImageAsArray(image, show_output=False)
        pred = self.final_model.predict(image_as_array)
        sorted_ranks = np.flip(np.argsort(pred[0]))
        max_index = np.argmax(pred)
        print(f"index argmax : {max_index}\n\
            pred : {pred}\n")
        prob = pred[0][max_index]
        names = self.class_names[max_index]
        return prob, names
        
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

        # maxclass_prob, maxclass_name = self.prep_predict_debug(image)
        # maxclass_res = maxclass_name + ' with value ' + str(maxclass_prob)
        # response = {
        #     'class': class_input,
        #     'prob': maxclass_res
        # }
        # return response

        try:
            predku, sorted_ranku = self.prep_predict(image)
            response_class = class_input
            response_prob = self.rules(predku, sorted_ranku, class_input)
            print(f"[KELAS][SELESAI PROSES]: {response_class} {response_prob}")
            # Rest of the code
            # index_max = 
            response = {
            'class': response_class,
            'prob': str(response_prob) + ' debug mode'
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
        # FIRST
        if class_input == 'carakan_ha':
            res_buff = self.take_class(pred, sorted_rank, class_input)
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_la')
            res.append(res_buff)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]

        elif class_input == 'carakan_na':
            res_buff = self.take_class(pred, sorted_rank, class_input)
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_ka')
            res.append(res_buff)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]

        elif class_input == 'carakan_ca':
            res_buff = self.take_class(pred, sorted_rank, class_input)
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_sa')
            res.append(res_buff)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]

        elif class_input == 'carakan_ra':
            res_buff = self.take_class(pred, sorted_rank, class_input)
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'sandhangan_e2')
            res.append(res_buff)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]
        # SECOND
        elif class_input == 'carakan_da':
            res_buff = self.take_class(pred, sorted_rank, class_input)
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_sa')
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_ya')
            res.append(res_buff)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]
        elif class_input == 'carakan_sa':
            res_buff = self.take_class(pred, sorted_rank, class_input)
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_ya')
            res.append(res_buff)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]

        # THIRD
        elif class_input == 'carakan_pa':
            res_buff = self.take_class(pred, sorted_rank, class_input)
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_la')
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_ya')
            res.append(res_buff)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]

        elif class_input == 'carakan_dha':
            res_buff = self.take_class(pred, sorted_rank, class_input)
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_ca')
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_ka')
            res.append(res_buff)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]

        # FOURTH
        elif class_input == 'carakan_ba':
            res_buff = self.take_class(pred, sorted_rank, class_input)
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_ta')
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_nya')
            res.append(res_buff)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]

        elif class_input == 'carakan_nga':
            res_buff = self.take_class(pred, sorted_rank, class_input)
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_ma')
            res.append(res_buff)
            res_buff = self.take_class(pred, sorted_rank, 'carakan_ba')
            res.append(res_buff)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]

        else:
            res_buff = self.take_class(pred, sorted_rank, class_input)
            res.append(res_buff)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]
