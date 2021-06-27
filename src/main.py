import cv2
import numpy as np
import time
from process import Image, Model, Ensembel


def main():

    # Create the models
    model1 = Model('../model/model1.h5')
    model2 = Model('../model/model2.h5')

    # Intialize Video camera
    camera = cv2.VideoCapture(0)
    camera.set(3, 1280)
    
    # Time managment
    st = time.time()
    tt = 1

    val = ""
    while True:
        
        # Reading the camera
        ret, frame = camera.read()
        frame = cv2.flip(frame, 1)
        #aspect = frame.shape[1] / frame.shape[0]

        # check the image in every 1 seconds
        if time.time() - st >= tt:
            img = frame
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            image = Image(img)
            processed_img = image.process()

            pred1 = model1.predict(processed_img)
            pred2 = model2.predict(processed_img)

            ensembel = Ensembel(pred1, pred2)
            val, acc = ensembel.weighted_avg(1, 0, threshold=.3)

            print(val, acc)
            tt += 1
        
        color = (0, 0, 255)
        if val == 'Mask':
            color = (0, 255, 0)
        cv2.putText(frame, val, (70, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, color, 6)
            
        #Showing the video on screen     
        cv2.imshow('Mask Detection', frame)

        #Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    camera.release() 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
