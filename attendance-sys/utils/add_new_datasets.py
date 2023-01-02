from numpy import asarray
from os import listdir
from os.path import isdir
from keras.models import load_model
from sklearn import datasets
from .helper_functions import get_embedding,extract_face

import albumentations as A
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from numpy import load, concatenate,savez_compressed
from numpy import where, delete
from flask import current_app as app
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

def train_model(trainX, trainy):
  # normalize input vectors
  in_encoder = Normalizer(norm="l2")
  trainX = in_encoder.transform(trainX)
  # label encode targets
  out_encoder = LabelEncoder()
  out_encoder.fit(trainy)
  trainy = out_encoder.transform(trainy)
  
  #fit model
  model = SVC(kernel="linear", probability=True)
  model.fit(trainX, trainy)
  # save the model to disk
  filename = f'model.sav'
  pickle.dump(model, open(filename, 'wb'))
  print("model updated successfully")


# load images and extract faces for all images in a directory
def load_faces(directory):
  faces = list()
  # enumerate files
  for filename in listdir(directory):
    # path
    path = directory + filename
    # get face
    face = extract_face(path)
    assert face.size, f"Failed to extract face from {path}"
    # store
    faces.append(face)
  return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_new_dataset(directory, trained_classes):
  X, y = list(), list()
  # enumerate folders, on per class
  for subdir in listdir(directory):
    if subdir in trained_classes:
      continue
    # path
    path = directory + subdir + "/"
    # skip any files that might be in the dir
    if not isdir(path):
      continue
    # load all faces in the subdirectory
    faces = load_faces(path)
    
    # create labels
    labels = [subdir for _ in range(len(faces))]
    # summarize progress
    print(">loaded %d examples for class: %s" % (len(faces), subdir))
    # store
    X.extend(faces)
    y.extend(labels)
  return asarray(X), asarray(y)

from os import listdir
from numpy import where
def get_deleted_classes(directory,trained_classes):
  current_dirs = set()
    # enumerate folders, on per class
  for subdir in listdir(directory):
    # path
    path = directory + subdir + "/"
    # skip any files that might be in the dir
    if not isdir(path):
          continue
    current_dirs.add(subdir)

  deleted_classes = trained_classes - current_dirs
  return deleted_classes


def get_new_embeddings(trainX):
      # convert each face in the train set to an embedding
    # load the facenet model
  model = load_model(f"facenet_keras.h5")
  print("Loaded Model")
  newTrainX = list()
  for face_pixels in trainX:
    embedding = get_embedding(model, face_pixels)
    newTrainX.append(embedding)
  newTrainX = asarray(newTrainX)
  print(newTrainX.shape)
  # convert each face in the test set to an embedding

  return newTrainX

def updated_train():
      
  datasets_dir = f"uploads/datasets/"

  # load dataset
  data = load(f"datasets-embeddings.npz")
  trainX, trainy= data["arr_0"], data["arr_1"]
  print(f"Dataset: train={trainX.shape} ")

  trained_classes = set(trainy)

    # load train dataset
  new_trainX, new_trainy = load_new_dataset(datasets_dir, trained_classes)
  print(new_trainX.shape, new_trainy.shape)
#datasets : 1 : 5 , 2: 5 , 3 : 5
# Augmentation code :  1 - 20, 2: 20, 3: 20

###############AUGMENTATION##############
path = "C:/Users/Dell/Desktop/attedance_pic/ATTENDANCE"

images = []
for filename in os.listdir(path):
  img = cv2.imread(os.path.join(path, filename))
  if img is not None:
    images.append(img)

#Data Augumentation: With Blurs and Distorsions
transform = A.Compose([
                       A.Transpose(p=0.5),
                       A.VerticalFlip(p=0.5),
                       A.HorizontalFlip(p=0.5),
                       A.Rotate(p=0.5),
                       A.RandomBrightness(limit=0.2, p=0.5),
                       A.RandomContrast(limit=0.2, p=0.5),
                       A.OneOf([
                                A.MotionBlur(blur_limit=5, p=0.25),
                                A.MedianBlur(blur_limit=5, p=0.25),
                                A.GaussianBlur(blur_limit=5, p=0.25),
                                A.GaussNoise(var_limit=(5.0, 30.0), p=0.25)                                
                       ]),
                       A.OneOf([
                                A.OpticalDistortion(distort_limit=1.0, p=0.25),
                                A.GridDistortion(num_steps=5, distort_limit=1., p=0.25),
                                A.ElasticTransform(alpha=3, p=0.25)                               
                       ]),
                       A.CLAHE(clip_limit=4.0, p=0.7),
                       A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
                       A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
                       A.Resize(width=722, height=542, p=0.5),
                       A.Normalize(max_pixel_value=255.0,p=0.5),
                       A.Cutout(max_h_size=int(h[i]*0.1), max_w_size=int(w[i]*0.1), num_holes=8, p=0.7)
                       ])

for img in images:
    transformed = transform(image=img)
    transformed_image = transformed["image"]

    cv2.imwrite('Augmented Sample Images', transformed_image)  


  # get new_embeddings 
  new_trainX = get_new_embeddings(new_trainX)
  print(new_trainX.shape, type(new_trainX))

  # concatenate trainX and new_trainX
  updated_trainX =trainX
  updated_trainy =trainy
  if new_trainX.size :
        # append 
    updated_trainX = concatenate((trainX, new_trainX))
    updated_trainy = concatenate((trainy, new_trainy))
  print("updated data: ",updated_trainX.shape, updated_trainy.shape)


  



  # delete 
  deleted_classes = get_deleted_classes(datasets_dir, trained_classes)
  for del_class in deleted_classes:
    indices_train = where(updated_trainy==del_class)[0]
    updated_trainX= delete(updated_trainX, indices_train, axis=0)
    updated_trainy= delete(updated_trainy, indices_train, axis=0)


    

   

  
  # update incase of any changes only 
  if  new_trainX.size or deleted_classes :
    # save compressed 
    savez_compressed(f"datasets-embeddings.npz", updated_trainX, updated_trainy)
    # retrain the model with updated data
    train_model(updated_trainX, updated_trainy)

if __name__ == "__main__":
  updated_train()