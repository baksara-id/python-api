from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import cv2
import numpy as np
from tensorflow import keras
from numpy import asarray
# self lib

from Scanner import Scanner
from Kelas import Kelas

app = Flask(__name__)
api = Api(app)

# Api endpoints
api.add_resource(Scanner, "/scanner")
api.add_resource(Kelas, "/kelas")

# Start server
PORT = 8080
if __name__ == '__main__':
    # Uncomment this on production
    # app.run(port=PORT)
    # Comment this on production
    app.run(debug=True, host='0.0.0.0', port=PORT)
    # app.run(debug=True, port=PORT)