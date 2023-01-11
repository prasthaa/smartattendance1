from numpy import asarray
from os import listdir
from os.path import isdir
from keras.models import load_model
from sklearn import datasets
from .helper_functions import get_embedding,extract_face

import albumentations as A
import random
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
      for img_name in os.listdir(os.path.join(path, filename)):
          img = cv2.imread(os.path.join(path, filename, img_name))
          if img is not None:
              images.append((img_name, img))

#Data Augumentation: With Blurs and Distorsions
  
  transform = A.Compose([
                        A.ImageCompression(p=1, quality_lower=30, quality_upper=60),
                        A.SmallestMaxSize(max_size=1024, interpolation=1, p=1),
                       
                    
                        A.OneOf([
                                A.HorizontalFlip(p=0.5),
                                A.Rotate(limit=90,p=0.5)
                        ],p=0.2),
                        A.OneOf([
                                A.RandomBrightness(limit=0.2, p=0.2),
                                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1),num_shadows_lower=1,num_shadows_upper=2,shadow_dimension=5,always_apply=False,p=0.2)
                        ],p=0.2),
                        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=False, p=0.4),
                    
                        A.OneOf([
                                A.CLAHE(clip_limit=4.0, always_apply=False, p=0.5),
                                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
                        ],p=0.2),
                        A.HueSaturationValue(hue_shift_limit=5,sat_shift_limit=8,val_shift_limit=8,p=0.5),
                        A.OneOf([
                                A.RGBShift(b_shift_limit=20,p=0.3), 
                                A.RGBShift(r_shift_limit=20,p=0.3),
                                A.RGBShift(g_shift_limit=20,p=0.3)
                        ],p=0.2),
                       
                        A.OneOf([
                                A.ToGray(p=0.2),
                                A.RandomGamma(gamma_limit=(20,45),p=0.2)
                        ],p=0.2),
                       
                        A.OneOf([
                                A.AdvancedBlur(blur_limit=(3,7), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0), rotate_limit=45, beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), always_apply=False, p=0.2),
                                A.Blur(blur_limit=3,p=0.2),
                                A.MotionBlur( p=0.2),
                                A.MedianBlur(blur_limit=3, p=0.2), 
                                A.GaussNoise(p=0.2),
                                A.GaussianBlur(blur_limit=3, p=0.2)
                        ], p=0.25),
                        
                        A.OneOf([
                                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5), 
                                A.GridDistortion(p=0.3),
                                A.ElasticTransform(alpha=1,sigma=50,alpha_affine=50,interpolation=1,border_mode=4,value=None,mask_value=None,always_apply=False,approximate=False,same_dxdy=False,p=0.5)
                        ],p=0.5),
                       
                        A.OneOf([
                                A.Perspective (scale=(0.05, 0.1), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False, interpolation=1, always_apply=False, p=0.5),
                                A.RandomScale(scale_limit=0.1,interpolation=1,always_apply=False, p=0.5)
                        ],p=0.5),
                        A.OneOf([
                                A.Transpose(p=0.5), 
                                A.ShiftScaleRotate(scale_limit=0.3, rotate_limit=45, shift_limit=0.3, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0)
                        ],p=0.3),
                        A.Resize(height=600, width=600,p=1),
                        A.Normalize(mean=(0.485, ), std=(0.229, ), p=0.05),
                        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.2),
                        A.OneOf([
                                A.RGBShift (r_shift_limit=60, g_shift_limit=20, b_shift_limit=20, p=0.3),
                                A.RGBShift (r_shift_limit=20, g_shift_limit=60, b_shift_limit=20, p=0.3),
                                A.RGBShift (r_shift_limit=0, g_shift_limit=0, b_shift_limit=60, p=0.3)
                        ],p=0.5),
                        A.OneOf([
                                #A.RandomSunFlare (flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, src_radius=400, src_color=(255, 255, 255), p=0.3),
                                A.RandomFog (fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.3),
                                A.RandomToneCurve (scale=0.1, p=0.3)
                        ],p=0.5),
                               
                        A.OneOf([
                               
                                A.ISONoise (color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
                                A.RandomBrightnessContrast (brightness_limit=-0.3, contrast_limit=0.2, brightness_by_max=True, p=0.3)
                        ],p=0.5),
                        A.Perspective (scale=(0.1, 0.2), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False, interpolation=1, p=0.1),
                        A.CoarseDropout (max_holes=12, max_height=8, max_width=8, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, p=0.2),
                    
                        A.OneOf([
                                A.ElasticTransform (alpha=1, sigma=50, alpha_affine=25, interpolation=1, border_mode=4, value=None, mask_value=None, approximate=False, same_dxdy=False, p=0.2),
                               
                                A.MotionBlur(blur_limit=7, allow_shifted=True, p=0.2)
                        ],p=0.5),    
                      
                        A.ColorJitter (brightness=0.1, contrast=0.2, saturation=0.4, hue=0.1, p=0.2)
                       
                          ])


  for img in images:
      transformed = transform(image=img)
      transformed_image = transformed["image"]

      cv2.imwrite('Augmented Sample Images', transformed_image)  
 
  for i in range(70):
    
      for img_name, img in images:
        
          transformed = transform(image=img)
          transformed_image = transformed["image"]
          cv2.imwrite('new_attendance_pictures/{}_{}.jpg'.format(img_name,i), transformed_image )
        
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