import random
import json
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import speech_recognition as sr
import pyttsx3

lemmatizer = WordNetLemmatizer()

intents_data = json.loads(open('intents.json').read()) 

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                return result
    else:
        return "Sorry, I couldn't understand your request."

def convert_text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def audio_bot():
    print("GO! MVM. is running!")
    while True:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak:")
            audio = recognizer.listen(source)

        try:
            user_input = recognizer.recognize_google(audio)
            print("You:", user_input)
            
            if user_input.lower() == "end":
                convert_text_to_speech("Goodbye!")
                break
            
            intents_result = predict_class(user_input)
            response = get_response(intents_result, intents_data)
            print("Bot:", response)
            convert_text_to_speech(response)

        except sr.UnknownValueError:
            print("Could not understand audio")
            convert_text_to_speech("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results: {0}".format(e))
            convert_text_to_speech("Could not request results")

if __name__ == "__main__":
    audio_bot()
