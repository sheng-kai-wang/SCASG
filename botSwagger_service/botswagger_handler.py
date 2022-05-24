import json
import jsonpath
from collections import defaultdict

# import module
from botSwagger_service.index_builder import IndexBuilder
from botSwagger_service.wordnet_increaser import WordnetIncreaser
from botSwagger_service.cosine_similarity_calculator import CosineSimilarityCalculator

# import WordNet
import nltk


class BotSwaggerHandler:
    def __init__(self) -> None:
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        self.JSONPATH_TO_NLU = '$..[x-input-template][*]'
        self.coupling_too_high_dict = defaultdict(list)

    def check_sentences(self, data) -> json:
        nlu_dict = self.__parse_botswagger(data)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        token_dict = IndexBuilder(nlu_dict, lemmatizer).get_token_dict()
        token_dict = WordnetIncreaser(token_dict, lemmatizer).get_updated_token_dict()

        for query_sentence_id in nlu_dict:
            print(str(query_sentence_id) + '======================================')
            similarity_dict = CosineSimilarityCalculator(token_dict).get_similarity_dict(nlu_dict, query_sentence_id)
            print(similarity_dict)
        # print(self.coupling_too_high_dict)
        return 'ok'

    def __parse_botswagger(self, data) -> dict:
        data = json.loads(data)
        nlu_data = jsonpath.jsonpath(data, self.JSONPATH_TO_NLU)
        nlu_dict = dict()
        for intent_id, intent in enumerate(nlu_data):
            for sentence_id, sentence in enumerate(intent):
                doc_id = str(intent_id+1) + '-' + str(sentence_id+1)
                nlu_dict[doc_id] = sentence
        return nlu_dict
