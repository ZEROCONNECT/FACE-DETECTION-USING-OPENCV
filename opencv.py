import cv2
import pickle


face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')
labels = {"image": 0}
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
with open("labels.pickle",'rb') as f:
   og_labels= pickle.load(f)
   labels={v:k for  k,v in og_labels.items()}

cap=cv2.VideoCapture(0)

while True:
    #capture frame-by-frame
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(grey, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in face:
     
        roi_grey= grey[y:y+h, x:x+w]  #(ycord_start, ycord_end)
        roi_color = frame[y:y + h, x:x + w]


     #recognize ? we can use deep learned model such as predict, keras, pytorch or scikit-learn

        id,conf= recognizer.predict(roi_grey)
        if conf >= 2 and conf <=85:
            print(id)
            print(labels[id])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id]
            color = [255,255,255]
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)





        img_item = "myimage.png"
        cv2.imwrite(img_item,roi_grey)
        end_cord_x=x+w
        end_cord_y=y+h


        color=(255,0,0) #BGR
        stroke =2
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y),color,stroke)

    #displaying the result
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#when everything done, release the capture
cap.destroyAllWindows()



