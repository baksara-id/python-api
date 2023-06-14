from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import cv2
from PIL import Image
import numpy as np
import BaksaraConst
from numpy import asarray


# API purpose
        # if 'image' not in request.files:
        #     return jsonify({'error': 'No image found'}), 400
        # file = request.files['image']
        # class_input = request.form['actual_class']
        # image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        


# class Kelas(Resource):
import cv2
import numpy as np

class Kelas:
    def __init__(self):
        self.class_names = BaksaraConst.CLASS_NAMES
        self.final_model = BaksaraConst.MODEL

    def post(self, image=None, inp_class=''):
        class_input = inp_class

        if image is None:
            raise ValueError("No image found")

        try:
            predku, sorted_ranku = self.prep_predict(image)
            response_class = class_input
            response_prob = self.rules(predku, sorted_ranku, class_input)

            print(f"[KELAS][SELESAI PROSES]: {response_class} {response_prob}")

            # Rest of the code
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

    def prep_predict_debug(self, image):
        image_as_array = self.PreprocessImageAsArray(image, show_output=False)
        pred = self.final_model.predict(image_as_array)
        sorted_ranks = np.flip(np.argsort(pred[0]))
        max_index = np.argmax(pred)
        return pred[max_index], self.class_names[max_index]
    def prep_predict(self, image):
        image_as_array = self.PreprocessImageAsArray(image, show_output=False)
        pred = self.final_model.predict(image_as_array)
        sorted_ranks = np.flip(np.argsort(pred[0]))
        print(f"\n\nprep_predict\n \
            pred\t: {pred}\n \
            sorted_ranks\t:{sorted_ranks}")
        max_index = np.argmax(pred)
        print(f"max index\t: {max_index}\n\
            max class\t: {self.class_names[max_index]}\n\n")
        return pred, sorted_ranks

    def rules(self, pred, sorted_rank, class_input):
        res = []
        if class_input == 'carakan_ha':
            res1 = self.take_class(pred, sorted_rank, 'carakan_ha')
            res.append(res1)
            res2 = self.take_class(pred, sorted_rank, 'carakan_ta')
            res.append(res2)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]

        elif class_input == 'pasangan_ra' or class_input == 'carakan_ra':
            res1 = self.take_class(pred, sorted_rank, 'pasangan_ra')
            res.append(res1)
            res2 = self.take_class(pred, sorted_rank, 'carakan_ra')
            res.append(res2)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]

        elif class_input == 'pasangan_ga' or class_input == 'carakan_ga':
            res1 = self.take_class(pred, sorted_rank, 'pasangan_ga')
            res.append(res1)
            res2 = self.take_class(pred, sorted_rank, 'carakan_ga')
            res.append(res2)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]

        elif class_input == 'pasangan_ya' or class_input == 'carakan_ya':
            res1 = self.take_class(pred, sorted_rank, 'pasangan_ya')
            res.append(res1)
            res2 = self.take_class(pred, sorted_rank, 'carakan_ya')
            res.append(res2)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]

        elif class_input == 'pasangan_nga' or class_input == 'carakan_nga':
            res1 = self.take_class(pred, sorted_rank, 'pasangan_nga')
            res.append(res1)
            res2 = self.take_class(pred, sorted_rank, 'carakan_nga')
            res.append(res2)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]
        
        elif class_input == 'sandhangan_e' or class_input == 'pasangan_wa':
            res1 = self.take_class(pred, sorted_rank, 'sandhangan_e')
            res.append(res1)
            res2 = self.take_class(pred, sorted_rank, 'pasangan_wa')
            res.append(res2)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]
        
        elif class_input == 'pasangan_dha':
            res1 = self.take_class(pred, sorted_rank, 'pasangan_dha')
            res.append(res1)
            res2 = self.take_class(pred, sorted_rank, 'pasangan_tha')
            res.append(res2)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]
        
        elif class_input == 'pasangan_ma' or class_input == 'sandhangan_ng':
            res1 = self.take_class(pred, sorted_rank, 'pasangan_ma')
            res.append(res1)
            res2 = self.take_class(pred, sorted_rank, 'sandhangan_ng')
            res.append(res2)
            highest_tuple = max(res, key=lambda x: x[1])
            return highest_tuple[1]


# Create an instance of the Kelas class
kelas_instance = Kelas()

# Load an image
image_path = "../87.jpg"
image = cv2.imread(image_path)

# Specify the input class
input_class = "carakan_ka"

# Call the post method
kelas_instance.post(image=image, inp_class=input_class)
