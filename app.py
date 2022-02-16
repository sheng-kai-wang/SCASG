from flask import Flask, request
from sentencesHandler import SentencesHandler

app = Flask(__name__)


@app.route('/', methods=['GET'])
def introduction():
    return "This is a service for generating RASA training sentences, based on wordnet and spaCy.<br>You can use it with /generateSentences"


@app.route('/generateSentences', methods=['POST'])
def generateSentences():
    return SentencesHandler().get_nlu(request.data)


if __name__ == '__main__':
    app.run('0.0.0.0', debug=False)