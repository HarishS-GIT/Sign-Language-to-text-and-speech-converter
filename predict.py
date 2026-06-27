import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from speech import speak


print("Loading Model...")

model = load_model("asl_model.h5")

print("Model Loaded Successfully")


labels = [
'A','B','C','D','E','F','G','H','I',
'J','K','L','M','N','O','P','Q','R',
'S','T','U','V','W','X','Y','Z',
'del','nothing','space'
]


# Sign to Word

word_map = {

    "H": "HELLO",
    "W": "WELCOME",
    "B": "BYE",
    "T": "THANK YOU",
    "Y": "YES",
    "N": "NO",
    "P": "PLEASE",
    "S": "SORRY",
    "F": "FOOD",
    "L": "LOVE",
    "A": "AGAIN",
    "C": "COME",
    "G": "GOOD",
    "M": "MORNING",
    "R": "READY"

}



sentence = ""

last_text = ""

last_time = 0

cooldown = 2


CONFIDENCE = 0.90



cap = cv2.VideoCapture(0)


print("Opening Camera...")


if not cap.isOpened():

    print("Camera not opened")
    exit()


print("Camera Opened Successfully")



while True:


    ret, frame = cap.read()


    if not ret:
        break



    # -------------------------
    # Hand Detection Area
    # -------------------------

    roi = frame[80:450,80:450]


    # Convert color
    roi_rgb = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2RGB
    )


    img = cv2.resize(
        roi_rgb,
        (64,64)
    )


    img = img.astype("float32") / 255.0


    img = np.expand_dims(
        img,
        axis=0
    )



    prediction = model.predict(
        img,
        verbose=0
    )



    class_id = np.argmax(prediction)

    confidence = np.max(prediction)


    text = labels[class_id]



    status = "No Hand Detected"



    if confidence >= CONFIDENCE:


        status = "Hand Detected"


        current_time = time.time()



        if text != last_text and current_time-last_time > cooldown:



            if text in word_map:


                sentence += word_map[text] + " "



            elif text == "space":

                sentence += " "



            elif text == "del":

                sentence = sentence[:-1]



            last_text = text

            last_time = current_time





    # -------------------------
    # Display
    # -------------------------


    cv2.rectangle(
        frame,
        (80,80),
        (450,450),
        (0,255,0),
        2
    )



    cv2.putText(
        frame,
        "Sign : " + text,
        (30,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )



    cv2.putText(
        frame,
        "Confidence : {:.2f}%".format(confidence*100),
        (30,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )



    cv2.putText(
        frame,
        status,
        (30,130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,0),
        2
    )



    # Bottom sentence box

    h,w,_ = frame.shape


    cv2.rectangle(
        frame,
        (20,h-100),
        (w-20,h-20),
        (255,255,255),
        -1
    )


    cv2.putText(
        frame,
        "Sentence : "+sentence,
        (30,h-50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )



    cv2.imshow(
        "Sign Language Translator",
        frame
    )



    key = cv2.waitKey(1)



    # Speak

    if key == ord('s'):

        print(
            "Speaking:",
            sentence
        )

        speak(sentence)



    # Clear

    if key == ord('c'):

        sentence = ""



    # Backspace

    if key == ord('b'):

        sentence = sentence[:-1]



    # Exit

    if key == 27:

        break




cap.release()

cv2.destroyAllWindows()
