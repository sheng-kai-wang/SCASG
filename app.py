from flask import Flask, request
from sentences_handler import SentencesHandler
from botSwagger_handler import botSwaggerHandler

app = Flask(__name__)


@app.route('/', methods=['GET'])
def introduction():
    return "This is a service for RASA-nlu and botSwagger processing, based on wordnet and spaCy.<br>If you want to generate RASA training sentences, you can use it with <b>\"/generateSentences\"</b>.<br>If you want to preprocess botSwagger, you can use it with <b>\"/preprocessBotSwagger\"</b>."


@app.route('/generateSentences', methods=['POST'])
def generateSentences():
    return SentencesHandler().get_nlu(request.data)


@app.route('/preprocessBotSwagger', methods=['POST'])
def preprocessBotSwagger():
    return botSwaggerHandler().check_sentences(request.data)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
