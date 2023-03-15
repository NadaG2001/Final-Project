from flask import Flask, url_for, render_template, request, jsonify, redirect
from numpy import array
import json
import random
import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import csv
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import model_from_json
from keras.utils import img_to_array
from PIL import Image
from io import BytesIO

nltk.download('popular')
lemmatizer = WordNetLemmatizer()

# loading our models (breast cancer model)
model = pickle.load(open("model.pkl", "rb"))

# load chatbot model
chat_model = load_model('Chat_Model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

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


@app.route('/customer_support')
def customer_support():
    return render_template('customer_support.html')


@app.route('/base')
def base():
    return render_template('index.html')

# for uploading medicine dataset


@app.route("/drugs_recommended")
def drugs_recommended():
    with open("medicine.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = list(reader)
    return render_template("drugs_recommended.html", header=header, data=data)


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


########################### chatbot  ###############################


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, chat_model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = chat_model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, chat_model)
    res = getResponse(ints, intents)
    return res


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    load_model()
    app.run(debug=True, threaded=False)

if __name__ == "app":
    load_model()

if __name__ == "__main__":
    app.run(debug=True)
