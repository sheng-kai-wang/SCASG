import json
import jsonpath
from collections import defaultdict

# import module
from botSwagger_service.index_builder import IndexBuilder
from botSwagger_service.wordnet_increaser import WordnetIncreaser
from botSwagger_service.cosine_similarity_calculator import CosineSimilarityCalculator
from botSwagger_service.spacy_increaser import SpacyIncreaser

# import WordNet
import nltk

# import spaCy
import spacy


class BotSwaggerHandler:
    def __init__(self) -> None:
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        self.JSONPATH_TO_NLU = '$..[x-input-template][*]'
        self.nlp = spacy.load("en_core_web_lg")

    def check_sentences(self, data) -> json:
        coupling_too_high_dict = defaultdict(list)
        nlu_dict = self.__parse_botswagger(data)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        token_dict = IndexBuilder(nlu_dict, lemmatizer).get_token_dict()
        token_dict = WordnetIncreaser(token_dict, lemmatizer, self.nlp).get_updated_token_dict()

        cosine_similarity_calculator = CosineSimilarityCalculator(token_dict, nlu_dict)
        spacy_increaser = SpacyIncreaser(nlu_dict, self.nlp)

        for query_sentence_id in nlu_dict:
            new_similarity_dict = self.__get_tfidf_spacy_similarity_dict(cosine_similarity_calculator, spacy_increaser, query_sentence_id)
            print(query_sentence_id, '===============================================')
            print('[new_similarity_dict]', new_similarity_dict)
            # 將相同 intent 的語句的相似度相加
            intent_sum_dict = defaultdict(lambda: 0.0)
            # current_intent_id = ''
            # times = 0
            for sentence_id in new_similarity_dict:
                # # 跳過自己
                # if sentence_id == query_sentence_id:
                #     print(sentence_id, 'continue ========================')
                #     continue
                intent_id = sentence_id.split('-')[0]
                # # 取算術平均
                # if intent_id != current_intent_id:
                #     current_intent_id = intent_id
                #     if times != 0:
                #         intent_sum_dict[str(int(intent_id)-1)] /= times
                #     times = 0
                # times += 1
                intent_sum_dict[intent_id] += new_similarity_dict[sentence_id]
            print('intent_sum_dict:', intent_sum_dict)
            # 如果有 intent 的相似度總和，高於查詢 intent 的相似度總和，則加到 coupling_too_high_dict 裡面
            query_intent_id = query_sentence_id.split('-')[0]
            query_intent_sum = intent_sum_dict[query_intent_id]
            for intent_id in intent_sum_dict:
                if intent_sum_dict[intent_id] > query_intent_sum:
                    # 找出 intent 中，是哪一句過高
                    max = 0.0
                    for sentence_id in new_similarity_dict:
                        if intent_id == sentence_id.split('-')[0]:
                            if new_similarity_dict[sentence_id] > max:
                                max = new_similarity_dict[sentence_id]
                    for sentence_id in new_similarity_dict:
                        if new_similarity_dict[sentence_id] == max:
                            coupling_too_high_dict[query_sentence_id].append(sentence_id)

        print('coupling_too_high_dict:', coupling_too_high_dict)
        return coupling_too_high_dict

    def __parse_botswagger(self, data) -> dict:
        data = json.loads(data)
        nlu_data = jsonpath.jsonpath(data, self.JSONPATH_TO_NLU)
        nlu_dict = dict()
        for intent_id, intent in enumerate(nlu_data):
            for sentence_id, sentence in enumerate(intent):
                doc_id = str(intent_id+1) + '-' + str(sentence_id+1)
                nlu_dict[doc_id] = sentence
        return nlu_dict

    def __get_tfidf_spacy_similarity_dict(self, cosine_similarity_calculator, spacy_increaser, query_sentence_id) -> dict:
        new_similarity_dict = dict()
        tfidf_similarity_dict = cosine_similarity_calculator.get_tfidf_similarity_dict(query_sentence_id)
        spacy_similarity_dict = spacy_increaser.get_spacy_similarity_dict(query_sentence_id)
        for sentence_id in tfidf_similarity_dict:
            new_similarity_dict[sentence_id] = (tfidf_similarity_dict[sentence_id] + spacy_similarity_dict[sentence_id]) / 2
        return new_similarity_dict
