import cv2 as cvision
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from time import sleep
from keras.preprocessing import image

#loading the trained data file


#face classification using haarcacade xml
cat_face = cvision.CascadeClassifier(r'C:\Users\preet\Downloads\hciMoss\hciMoss\Facial-Expressions-Recognition\haarcascade_frontalface_default.xml')
categorization =load_model(r'C:\Users\preet\Downloads\Emotion_little_vgg.h5')
#emotions categorized
outcome = ['Angry','Happy','Neutral','Sad','Surprise']

#capturing the live events
live = cvision.VideoCapture(0)

def frame_check(reshaped):
    if np.sum([reshaped]) == 0:
        cvision.putText(frame, 'No Face Found In The Frame', (20, 60), cvision.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        shape = reshaped.astype('float') / 255.0
        shape = img_to_array(shape)
        shape = np.expand_dims(shape, axis=0)
        result_pred = categorization.predict(shape)[0]
        icon = outcome[result_pred.argmax()]
        print(icon)
        f = open("faceexpressions.txt", "a")
        f.write(icon)
        f.write("\n")
        f.close()
        icon_pos = (x, y)
        cvision.putText(frame, icon, icon_pos, cvision.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)


def dummy(x,y,w,h) :
     cvision.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
     reshaped = picture[y:y + h, x:x + w]
     reshaped = cvision.resize(reshaped, (48, 48), interpolation=cvision.INTER_AREA)
     return reshaped


while True:
    # Grab a single frame of video
    isTrue, frame = live.read()
    # labels = []
    picture = cvision.cvtColor(frame, cvision.COLOR_BGR2GRAY)
    found_faces = cat_face.detectMultiScale(picture, 1.3, 5)

    for (x, y, w, h) in found_faces:
        reshaped = dummy(x,y,w,h)
        frame_check(reshaped)

    cvision.imshow('Emotion finder', frame)
    if cvision.waitKey(1) & 0xFF == ord('q'):
        break      

live.release()
cvision.destroyAllWindows()



























