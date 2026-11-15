import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch
# In the below line, remove '.' while working on your local system. However Make sure that '.' is present before face_recognition_model while uploading to the server, Do not remove it.
from .face_recognition_model import *
from PIL import Image
import base64
import io
import os
import joblib
import pickle
import json

###########################################################################################################################################
#         Caution: Don't change any of the filenames, function names and definitions                                                      #
#        Always use the current_path + file_name for refering any files, without it we cannot access files on the server                  #
###########################################################################################################################################

# Current_path stores absolute path of the file from where it runs.
current_path = os.path.dirname(os.path.abspath(__file__))

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Load both models once at import time (not per-request) so each upload is fast ---

_siamese_model = Siamese().to(_device)
_siamese_ckpt = torch.load(current_path + '/siamese_model.t7', map_location=_device)
_siamese_model.load_state_dict(_siamese_ckpt['net_dict'])
_siamese_model.eval()

_classifier = joblib.load(current_path + '/decision_tree_model.sav')

_class_map_path = current_path + '/class_mapping.json'
if os.path.exists(_class_map_path):
    with open(_class_map_path) as f:
        _idx_to_class = {int(k): v for k, v in json.load(f).items()}
else:
    # Fallback: use the order in face_recognition_model.classes
    _idx_to_class = {i: c for i, c in enumerate(classes)}


#1) The below function is used to detect faces in the given image.
#2) It returns only one image which has maximum area out of all the detected faces in the photo.
#3) If no face is detected,then it returns zero(0).

def detected_face(image):
    eye_haar = current_path + '/haarcascade_eye.xml'
    face_haar = current_path + '/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_haar)
    eye_cascade = cv2.CascadeClassifier(eye_haar)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    face_areas = []
    images = []
    required_image = 0
    for i, (x, y, w, h) in enumerate(faces):
        face_cropped = gray[y:y+h, x:x+w]
        face_areas.append(w*h)
        images.append(face_cropped)
        required_image = images[np.argmax(face_areas)]
        required_image = Image.fromarray(required_image)
    return required_image


#1) Images captured from mobile is passed as parameter to the below function in the API call. It returns the similarity measure between given images.
#2) The image is passed to the function in base64 encoding, Code for decoding the image is provided within the function.
#3) Define an object to your siamese network here in the function and load the weight from the trained network, set it in evaluation mode.
#4) Get the features for both the faces from the network and return the similarity measure, Euclidean,cosine etc can be it. But choose the Relevant measure.
#5) For loading your model use the current_path+'your model file name', anyhow detailed example is given in comments to the function
#Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_similarity(img1, img2):
    det_img1 = detected_face(img1)
    det_img2 = detected_face(img2)
    if isinstance(det_img1, int) or isinstance(det_img2, int):
        det_img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
        det_img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

    face1 = trnscm(det_img1).unsqueeze(0).to(_device)
    face2 = trnscm(det_img2).unsqueeze(0).to(_device)

    with torch.no_grad():
        output1, output2 = _siamese_model(face1, face2)
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)

    return float(euclidean_distance.item())


#1) Image captured from mobile is passed as parameter to this function in the API call, It returns the face class in the string form ex: "Person1"
#2) The image is passed to the function in base64 encoding, Code to decode the image provided within the function
#3) Define an object to your network here in the function and load the weight from the trained network, set it in evaluation mode
#4) Perform necessary transformations to the input(detected face using the above function).
#5) Along with the siamese, you need the classifier as well, which is to be finetuned with the faces that you are training
##Caution: Don't change the definition or function name; for loading the model use the current_path for path example is given in comments to the function
def get_face_class(img1):
    det_img1 = detected_face(img1)
    if isinstance(det_img1, int):
        det_img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))

    face1 = trnscm(det_img1).unsqueeze(0).to(_device)

    with torch.no_grad():
        embedding = _siamese_model.forward_once(face1).cpu().numpy()

    pred_idx = int(_classifier.predict(embedding)[0])
    return _idx_to_class.get(pred_idx, f"class_{pred_idx}")
