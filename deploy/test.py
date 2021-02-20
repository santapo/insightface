import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--gpu', default=-1, type=int, help='gpu id')
args = parser.parse_args()

vec = args.model.split(',')
model_prefix = vec[0]
model_epoch = int(vec[1])
model = face_model.FaceModel(args.gpu, model_prefix, model_epoch)
img = cv2.imread('MicrosoftTeams-image.png')
img = model.get_input(img)

img2 = cv2.imread('img1.png')
img2 = model.get_input(img2)

f1 = model.get_feature(img)
f2 = model.get_feature(img2)
sim = np.dot(f1, f2)
print(sim)
assert(sim>=0.99 and sim<1.01)
