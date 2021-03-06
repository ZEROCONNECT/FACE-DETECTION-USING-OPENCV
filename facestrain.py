import  os
import cv2

from PIL import Image
import pickle
import numpy as np


base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir,"image",)

face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids={}
y_labels= []
x_train = []



for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).lower() # os.path.dirname(path) = root
            #print(label,path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id = label_ids[label]
            #print(label_ids)

            #y_labels.append(label) # some no. for label
           # x_train.append(path) # verify this image, turn into a NUMPY array, GRAY
            pill_image = Image.open(path).convert('L') #greyscale

            image_array = np.array(pill_image,"uint8")
            print(image_array)

            faces= face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h,x:x+h]
                x_train.append(roi)
                y_labels.append(id)

# print(y_labels)
# print(x_train)

with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
