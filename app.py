from flask import Flask, request
from sentence_service.sentences_handler import SentencesHandler
from botSwagger_service.botswagger_handler import BotSwaggerHandler

app = Flask(__name__)

sentences_handler = SentencesHandler()
botswagger_handler = BotSwaggerHandler()


@app.route('/', methods=['GET'])
def introduction():
    return "This is a service for RASA-nlu and botSwagger processing, based on WordNet and spaCy.<br>If you want to generate RASA training sentences, you can use it with <b>\"/generateSentences\"</b>.<br>If you want to preprocess botSwagger, you can use it with <b>\"/preprocessBotSwagger\"</b>."


@app.route('/generateSentences', methods=['POST'])
def generateSentences():
    return sentences_handler.get_nlu(request.data)


@app.route('/preprocessBotSwagger', methods=['POST'])
def preprocessBotSwagger():
    return botswagger_handler.check_sentences(request.data)


if __name__ == '__main__':
    app.run(port=5000, debug=False)
