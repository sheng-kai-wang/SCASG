from flask import Flask, request
from sentences_handler import SentencesHandler
from botSwagger_handler import botSwaggerHandler

app = Flask(__name__)


@app.route('/', methods=['GET'])
def introduction():
    return "This is a service for RASA-nlu and botSwagger processing, based on wordnet and spaCy.<br>If you want to generate RASA training sentences, You can use it with /generateSentences.<br>If you want to preprocess botSwagger, You can use it with /preprocessBotSwagger."


@app.route('/generateSentences', methods=['POST'])
def generateSentences():
    return SentencesHandler().get_nlu(request.data)


@app.route('/preprocessBotSwagger', methods=['POST'])
def generateSentences():
    return botSwaggerHandler().check_sentences(request.data)


if __name__ == '__main__':
    app.run('0.0.0.0', debug=False)