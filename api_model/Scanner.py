from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import cv2
import numpy as np
from tensorflow import keras
from numpy import asarray

# self defined
import BaksaraConst



class Scanner(Resource):
    ''' 
        url : {base_url}/scanner
        used for : Class for scanning feature in Baksara       
    '''

    def post(self):
        ''' 
            url : {base_url}/scanner | Method:POST
        '''
        # Check if an image file is present in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image found'}), 400

        file = request.files['image']

        # Read the image file
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)  

        # Perform segmentation
        segmentation_result = self.segmentation(image)
        # Perform classification
        classification_result = self.classification(segmentation_result)
        # Perform transliteration
        transliteration_result = self.transliteration(classification_result)

        final_result = []
        for i, res in enumerate(transliteration_result):
            joined_result = ' '.join(res)
            final_result.append(joined_result)
        response = jsonify(result = final_result)
        return response

    def dilate_image(self, image, kernel_size):
        # Set the kernel size and sigma
        sigma = 2
        # Define the kernel for dilation
        kernel = np.ones((kernel_size, sigma), np.uint8)
        # Perform dilation
        dilated_image = cv2.dilate(image, kernel, iterations=1)
        return dilated_image

    def horizontal_pp(self, target_image, input_image, threshold):
        h_projection = np.sum(input_image, axis=1)

        start_putih = []
        end_putih = []

        h_segmentation = []
        for index, value in enumerate(h_projection):
            if value > threshold:
                if len(start_putih) == len(end_putih):
                    start_putih.append(index)
            else:
                if len(start_putih) == (len(end_putih) + 1):
                    end_putih.append(index)

        if len(start_putih) == len(end_putih) + 1:
            end_putih.append(len(h_projection))

        for start, end in zip(start_putih, end_putih):
            segmented_image = target_image[start:end, :]
            h_segmentation.append(segmented_image)

        return h_segmentation

    def vertical_pp(self, image):
        v_projection = np.sum(image, axis=0)

        start_putih = []
        end_putih = []

        v_segmentation = []
        for index, value in enumerate(v_projection):
            if value > 0:
                if len(start_putih) == len(end_putih):
                    start_putih.append(index)
            else:
                if len(start_putih) == (len(end_putih) + 1):
                    end_putih.append(index)

        if len(start_putih) == len(end_putih) + 1:
            end_putih.append(len(v_projection))

        for start, end in zip(start_putih, end_putih):
            segemented_image = image[:, start:end]
            v_segmentation.append(segemented_image)

        return v_segmentation

    def image_to_canfas(self, image):
        # Calculate the new size with aspect ratio preserved
        offset = 18
        max_size = 128 - 2 * offset
        height, width = image.shape[:2]

        if height > width:
            new_height = max_size
            ratio = new_height / height
            new_width = int(width * ratio)
        else:
            new_width = max_size
            ratio = new_width / width
            new_height = int(height * ratio)

        # Resize the image with the calculated size
        resized_image = cv2.resize(image, (new_width, new_height))
        # Create the canvas with padding
        canvas = np.zeros((128, 128), dtype=np.uint8)
        # calculate the x middle
        # 128 / 2 = 64
        x_start = 64-new_width//2
        y_start = 64-new_height//2
        canvas[y_start:y_start+new_height,
            x_start:x_start+new_width] = resized_image
        return canvas

    def segmentation(self, input_image):
        # Perform segmentation using horizontal_pp method
        threshold_image = input_image  # Replace with your thresholding logic
        dilated_image = self.dilate_image(threshold_image, 41)
        h_segmentation = self.horizontal_pp(threshold_image, dilated_image, 0)

        # ... continue with the rest of the segmentation logic
        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        # Convert Grayscale into Binary Image
        _, threshold_image = cv2.threshold(
            gray_image, 150, 255, cv2.THRESH_BINARY_INV)
        dilated_image = self.dilate_image(threshold_image, 41)

        # FIRST HORIZONTAL PP
        h1_segmentation = self.horizontal_pp(threshold_image, dilated_image,  0)
        # Remove empty segments at the beginning and end (if any)
        h1_segmentation = [
            region for region in h1_segmentation if np.sum(region) > 0]

        # FIRST VERTICAL PP
        h1_regions = len(h1_segmentation)
        v1_segmentation = []
        for i in range(h1_regions):
            v_segmentation = self.vertical_pp(h1_segmentation[i])
            # Remove empty segments at the beginning and end (if any)
            v_segmentation = [
                region for region in v_segmentation if np.sum(region) > 0]

            v1_segmentation.append(v_segmentation)

        # SECOND HORIZONTAL PP
        v1_regions = len(v1_segmentation)
        h2_segmentation = []
        tebal_px = 13
        h2_treshold = int(255 * tebal_px)
        for i in range(v1_regions):
            num_images = len(v1_segmentation[i])
            h2_temp = []

            for j in range(num_images):
                h_segmentation = self.horizontal_pp(
                    v1_segmentation[i][j], v1_segmentation[i][j], h2_treshold)
                # Remove empty segments at the beginning and end (if any)
                h_segmentation = [
                    region for region in h_segmentation if np.sum(region) > 0]

                h2_temp.append(h_segmentation)
            h2_segmentation.append(h2_temp)

        # SECOND VERTICAL PP
        h2_regions = len(h2_segmentation)
        v2_segmentation = []

        for i in range(h2_regions):
            num_col = len(h2_segmentation[i])
            # Menyimpan per baris
            v2_temp = []

            for j in range(num_col):
                num_img = len(h2_segmentation[i][j])
                if num_img == 1:
                    v2_temp.append(h2_segmentation[i][j])

                else:
                    # Menyimpan per satu set aksara dalam satu baris yang sama
                    v2_temp_2 = []
                    for k in range(num_img):
                        v_segmentation = self.vertical_pp(h2_segmentation[i][j][k])
                        # Remove empty segments at the beginning and end (if any)
                        v_segmentation = [
                            region for region in v_segmentation if np.sum(region) > 0]

                        if len(v_segmentation) > 1:
                            v2_temp.append([v_segmentation[0]])
                            v2_temp_2.append(v_segmentation[1])
                        else:
                            v2_temp_2.append(v_segmentation[0])

                    if (len(v2_temp_2) != 0):
                        v2_temp.append(v2_temp_2)
            v2_segmentation.append(v2_temp)

        # Image to canvas function
        

        # Set each image in the center of 128x128 canvas and perform bitwise not
        v2_regions = len(v2_segmentation)
        for i in range(v2_regions):
            num_col = len(v2_segmentation[i])
            for j in range(num_col):
                num_img = len(v2_segmentation[i][j])
                for k in range(num_img):
                    v2_segmentation[i][j][k] = cv2.bitwise_not(
                        self.image_to_canfas(v2_segmentation[i][j][k]))

        return v2_segmentation

    def PreprocessInput(self, image):
        # Resize the image to match the model's input shape
        image = cv2.resize(image, (128, 128))
        # Add batch dimension
        image_as_array = np.expand_dims(asarray(image), axis=0)
        scaled_image_as_array = np.true_divide(image_as_array, 255)
        return scaled_image_as_array

    def classification(self, array_images):
        # Load Model
        final_model = BaksaraConst.MODEL

        # Define Class Names
        class_names = BaksaraConst.CLASS_NAMES

        # Preprocess Function
        

        result = []

        # Display the number of segmented regions
        regions = len(array_images)
        for i in range(regions):
            num_col = len(array_images[i])
            result_col = []
            num_images = 0
            for j in range(num_col):
                num_img = len(array_images[i][j])
                num_images += num_img

            if num_images == 1:
                input_image = array_images[i][0][0]
                # Convert Grayscale to RGB
                if len(input_image.shape) == 2:
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
                input_image = self.PreprocessInput(input_image)
                # Prediction
                pred = final_model.predict(input_image)
                sorted_ranks = np.flip(np.argsort(pred[0]))
                result_col.append(class_names[sorted_ranks[0]])
            else:
                for j in range(num_col):
                    result_imgs = []
                    num_img = len(array_images[i][j])

                    for k in range(num_img):
                        input_image = array_images[i][j][k]
                        # Convert Grayscale to RGB
                        if len(input_image.shape) == 2:
                            input_image = cv2.cvtColor(
                                input_image, cv2.COLOR_GRAY2BGR)
                        input_image = self.PreprocessInput(input_image)
                        # Prediction
                        pred = final_model.predict(input_image)
                        sorted_ranks = np.flip(np.argsort(pred[0]))
                        result_imgs.append(class_names[sorted_ranks[0]])
                    if (len(result_imgs) > 0):
                        result_col.append(result_imgs)

            result.append(result_col)
        return result
        
    def transliteration(pred_result):
        final_result = []
        num_row = len(pred_result)
        for i in range(num_row):
            row_result = []
            probably_o = False
            temp_o = []
            num_col = len(pred_result[i])
            for j in range(num_col):
                num_img = len(pred_result[i][j])
                for k in range(num_img):
                    jenis, aksara = pred_result[i][j][k].split('_')
                    if jenis == 'carakan':
                        if probably_o:
                            if (len(temp_o) == 1):
                                temp_o.append(aksara)
                            elif (len(temp_o) == 2):
                                # Aksara using sandangan e not o
                                row_result.append(temp_o[1][:-1] + temp_o[0])
                                probably_o = False
                                temp_o = []
                                row_result.append(aksara)
                            elif (len(temp_o) == 3) and (temp_o[-1] in ['h', 'ng', 'r']):
                                # sandhangan è using 'h', 'ng', 'r'
                                row_result.append(
                                    temp_o[1][:-1] + temp_o[0] + temp_o[2])
                                probably_o = False
                                temp_o = []
                                row_result.append(aksara)
                            else:
                                row_result.append(aksara)
                        else:
                            row_result.append(aksara)
                    elif jenis == 'sandhangan':
                        # taling 'è'
                        if aksara == 'e2':
                            probably_o = True
                            temp_o.append('è')
                        # tarung 'o'
                        elif aksara == 'o':
                            if probably_o:
                                if (len(temp_o) == 2):
                                    row_result.append(temp_o[1][:-1] + aksara)
                                    probably_o = False
                                    temp_o = []
                            # Terdeteksi o namun tidak ada taling, kemungkinan missclasify h
                            else:
                                row_result[-1] = row_result[-1] + 'h'
                        # e, i, u
                        elif aksara in ['e', 'i', 'u']:
                            row_result[-1] = row_result[-1][:-1] + aksara
                        # h, ng, r
                        else:
                            if (probably_o):
                                if (len(temp_o) == 2):
                                    temp_o.append(aksara)
                            else:
                                row_result[-1] = row_result[-1] + aksara
            final_result.append(row_result)
        return final_result
