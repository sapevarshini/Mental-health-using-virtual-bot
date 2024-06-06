import random
import json
import pickle
import numpy as np
import nltk
import pyttsx3
import speech_recognition as sr

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


model = load_model('chatbot_model.keras')


words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


recognizer = sr.Recognizer()


engine = pyttsx3.init()

def predict_class(sentence):
    # Preprocess the input
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [WordNetLemmatizer().lemmatize(word.lower()) for word in sentence_words]
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    
    result = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

def speak(text):
    engine.say(text)
    engine.runAndWait()

print("GO! Bot. is running!")

while True:

    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
    
        print("Recognizing...")
        user_input = recognizer.recognize_google(audio)
        print("You:", user_input)


        predicted_intent = predict_class(user_input)

        response = get_response(predicted_intent, intents)
        print("Bot:", response)

        speak(response)

    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
