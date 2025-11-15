### Prodigy InfoTech ML Internship: Task 4

This repository contains the completed Task 4 for the Prodigy InfoTech Machine Learning Internship.

Task: "Develop a hand gesture recognition model that can accurately identify and classify different hand gestures..."

### 🚀 Live Demo App

You can try the live gesture recognition app I built with Streamlit. It uses your webcam to classify your hand gestures in real-time.

Heads-up: As you'll read in my analysis below, this model is a "lab model" and fails hilariously when exposed to real-world webcam images. This failure is the entire point!

🔗 Live App URL: https://vishwash-ml-task-02.streamlit.app/

### 🧠 My "Knowledge Gained": A Journey in Failure

This task was a masterclass in the real-world challenges of machine learning. My goal was to build an accurate model, but I discovered that "accuracy" is a very misleading word.

Here is the step-by-step journey:

Attempt 1: The Overfit Model

I first trained a CNN on 2,000 images from one person and tested it on a new person. The result was a classic, massive overfit.

- Train Accuracy: 98.4% (It memorized Person 00)

- Test Accuracy: 37.9% (It was useless on Person 01)

- Lesson: A model trained on non-diverse data can't generalize.

Attempt 2: The "Accurate" Lab Model

I fixed the overfitting by creating a diverse dataset. I re-trained the model on 18,000 images from 9 different people, testing on 2,000 images from 1 new person.

- Train Accuracy: 97.2%

- Test Accuracy: 89.65%

This was a huge success! I had an "accurate" model... or so I thought.

Attempt 3: The "Real-World" Failure (The Bug Hunt)

When I deployed this "accurate" 89.65% model to my Streamlit app, it failed spectacularly. It predicted "L-Shape" for every gesture with 100% confidence.

This led to a long bug hunt where I discovered the real problem:

- The Bug: A Normalization Mismatch.

- My trainer (```task_04.py```) was training on pixels from ```[0, 255]```.

- My app (```app.py```) was normalizing them to ```[0, 1]```.

- The model was trained on one "language" and was being fed another.

The Final, "Correct" Model (It Still Fails!)

I fixed the normalization bug, re-trained the model (```hand_gesture_model_FINAL.h5```), and re-deployed. The app finally started making predictions.

And it was still terrible.

As you can see from my tests, it gets some right ("Fist", "Index") but fails on others ("Palm" -> "Down", "OK" -> "L-Shape"). This is after mimicking the dark environment of the training data.

### conclusion: The Most Important Lesson (Domain Mismatch)

This is the final and most important lesson of the entire internship.

The model is 89.65% accurate, but only at classifying other perfect lab images from the ```leapGestRecog``` dataset.

That dataset was created with a special infrared (Leap Motion) camera on a perfect black background. My model didn't learn "what a hand is"; it learned "what a high-contrast, glowing-white hand shape looks like on a black void."

My webcam, even in a dark room, is a completely different "domain." It has complex shadows, different lighting, and a real-world background. The "Domain Mismatch" is simply too large for the model to handle.

The "Knowledge Gained" is this:
A model's "accuracy" is a meaningless number if the training data doesn't match the real-world environment. The real task of an ML Engineer isn't just ```model.fit()```, it's closing this "domain gap" with better data (e.g., from real webcams) or advanced Data Augmentation (like pasting hands onto random backgrounds).

### 📂 Files in this Repository

- ```task_04.py``` : The FINAL "Trainer" script. It loads 18,000 images, applies the correct normalization, and trains the hand_gesture_model_FINAL.h5 file.

- ```app.py```: The FINAL "Inference" script. This is the Streamlit app that loads the final model and applies the correct (and matching) normalization to the webcam feed.

- ```hand_gesture_model_FINAL.h5```: The final, 89.65%-accurate "lab" model.

- ```requirements.txt```: All Python libraries needed.

- ```packages.txt```: Tells Streamlit Cloud to install Linux libraries for ```opencv-python```.

- ```.python-version```: Tells Streamlit Cloud to use Python ```3.11```.

- ```.gitignore```: The critical file that ignores the ```venv/``` and ```data/``` folders (20,000+ images).

### 🏃 How to Run This Project

Clone the repository:
```
git clone https://github.com/Redoxftw/PRODIGY_ML_04.git
cd PRODIGY_ML_04
```

Create and activate a virtual environment:
```
py -3.11 -m venv venv
.\venv\Scripts\activate
```

Install the required libraries:
```
pip install -r requirements.txt
```

Run the interactive Streamlit app:
(You don't need to run task_04.py since the hand_gesture_model_FINAL.h5 is already included!)
```
streamlit run app.py
```