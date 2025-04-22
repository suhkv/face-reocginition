from flask import Flask, render_template, Response, request
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load pre-trained model
model = load_model("FER_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

emotion_feedback = {
    "Happy": "You might be feeling joyful or satisfied—maybe something good just happened! 😊",
    "Sad": "You may be upset or missing something important. It's okay to feel this way sometimes. 😔",
    "Angry": "You might be frustrated or facing something unfair. Take a deep breath. 😡",
    "Surprise": "Something unexpected probably caught your attention. 😲",
    "Fear": "You could be facing an uncertain or risky situation. You're not alone. 😨",
    "Disgust": "Maybe you saw or experienced something unpleasant. 😒",
    "Neutral": "You seem calm or focused. Everything okay? 😐"
}

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect_emotion():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Camera error"

    cv2.imwrite("static/capture.jpg", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion_result = "No face detected"
    feedback = "Try again with a clearer view."

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        prediction = model.predict(roi)[0]
        maxindex = int(np.argmax(prediction))
        emotion_result = emotion_labels[maxindex]
        feedback = emotion_feedback[emotion_result]
        break

    return render_template("index.html", emotion=emotion_result, feedback=feedback)

if __name__ == "__main__":
    app.run(debug=True)
