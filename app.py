import time
import openai
from flask import Flask, render_template, request
from flask import Flask, url_for, render_template, request, jsonify, redirect
from numpy import array
import json
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow
from keras.models import load_model
from keras.models import model_from_json
from keras.models import load_model
import nltk
import csv
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import model_from_json
from keras.utils import img_to_array
from PIL import Image
from io import BytesIO

nltk.download('popular')

# loading our models (breast cancer model)
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__, template_folder="templates")


@app.route('/')
def website():
    return render_template('WellBeing.html')


@app.route('/test')
def test():
    return render_template('result.html')


@app.route('/disease_info')
def disease_info():
    return render_template('disease_info.html')


@app.route('/base')
def base():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    request.form.values()
    formValues = [j for j in request.form.values()]
    name = formValues[0]
    print(name)
    formValues.pop(0)
    floatFeatures = [float(j) for j in formValues]
    features = [np.array(floatFeatures)]
    prediction = int(model.predict(features))

    if prediction == 1:
        prediction_text = 'Malignant'
    elif prediction == 0:
        prediction_text = 'Benign'

    return render_template('result.html', prediction_text='The Tumor is {}'.format(prediction_text))

########################### Chest X-Rays  ###############################


def load_model():
    # open file with model architecture
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    global model_chest
    model_chest = model_from_json(loaded_model_json)

    # load weights into new model
    model_chest.load_weights("model.h5")
    print(model_chest.summary())


def process_image(image):
    # read image
    image = Image.open(BytesIO(image))
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize and convert to tensor
    image = image.resize((96, 96))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/", methods=["POST", "GET"])
def index():
    predictions = {}
    if request.method == "POST":
        # only make predictions after sucessfully receiving the file
        if request.files:
            try:
                image = request.files["image"].read()
                image = process_image(image)
                out = model_chest.predict(image)
                # send the predictions to index page
                predictions = {"positive": str(
                    np.round(out[0][1], 2)), "negative": str(np.round(out[0][0], 2))}
            except:
                predictions = {}
                redirect('/')
    return render_template("index.html", predictions=predictions)


########################### chatbot GPT-3  ###############################

openai.api_key = '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'

data = {
    "greetings": [
        "Hi, how can I help you today?",
        "Hello there! What can I do for you?",
        "Hey, what's up?",
        "Greetings! How may I assist you?",
        "Good day! How can I be of service?",
    ],
    "farewells": [
        "Goodbye! Have a great day.",
        "Farewell! See you soon.",
        "Take care!",
        "Bye! Come back anytime.",
        "Have a great day ahead!",
    ],
}


def bot(prompt, engine='text-davinci-002', temperature=0.9, max_tokens=500, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.9, stop=[" User:", " AI:"]):
    try:
        if "hi" in prompt.lower():
            response = openai.Completion.create(
                engine=engine,
                prompt=data["greetings"],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
        elif "bye" in prompt.lower():
            response = openai.Completion.create(
                engine=engine,
                prompt=data["farewells"],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
        else:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
        text = response.choices[0].text.strip()
        return text
    except Exception as e:
        return "GPT-3 Error: {}".format(e)


@app.route("/gpt_bot")
def gpt_bot():
    return render_template("gpt_bot.html")


@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    start_time = time.time()
    bot_response = bot(prompt=user_text)
    response_time = time.time() - start_time
    return str(bot_response)


###########################################################################

if __name__ == "__main__":
    load_model()
    app.run(debug=True, threaded=False)

if __name__ == "app":
    load_model()

if __name__ == "__main__":
    app.run(debug=True)
