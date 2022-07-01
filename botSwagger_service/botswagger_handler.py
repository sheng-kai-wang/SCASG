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
        nlu_dict = self._parse_botswagger(data)
        # print('[nlu_dict]', nlu_dict)
        lemmatizer = nltk.stem.WordNetLemmatizer()
        token_dict = IndexBuilder(nlu_dict, lemmatizer).get_token_dict()
        original_token_dict = token_dict.copy() # 內容複製一份給其他變數存
        # print('[original_token_dict]', original_token_dict)
        updated_token_dict = WordnetIncreaser(token_dict, lemmatizer, self.nlp).get_updated_token_dict()

        # print('[updated_token_dict]', updated_token_dict)
        cosine_similarity_calculator = CosineSimilarityCalculator(original_token_dict, updated_token_dict, nlu_dict)
        spacy_increaser = SpacyIncreaser(nlu_dict, self.nlp)

        self._show_average_similarity_dict(nlu_dict, cosine_similarity_calculator, spacy_increaser)

        for query_sentence_id in nlu_dict:
            new_similarity_dict = self._get_tfidf_spacy_similarity_dict(cosine_similarity_calculator, spacy_increaser, query_sentence_id)
            print(query_sentence_id, '===============================================')
            # print('[new_similarity_dict]', new_similarity_dict)
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
            # print('intent_sum_dict:', intent_sum_dict)
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

        # print('coupling_too_high_dict:', coupling_too_high_dict)
        return coupling_too_high_dict

    def _parse_botswagger(self, data) -> dict:
        data = json.loads(data)
        nlu_data = jsonpath.jsonpath(data, self.JSONPATH_TO_NLU)
        nlu_dict = dict()
        for intent_id, intent in enumerate(nlu_data):
            for sentence_id, sentence in enumerate(intent):
                doc_id = str(intent_id+1) + '-' + str(sentence_id+1)
                nlu_dict[doc_id] = sentence
        return nlu_dict

    def _get_tfidf_spacy_similarity_dict(self, cosine_similarity_calculator, spacy_increaser, query_sentence_id) -> dict:
        new_similarity_dict = dict()
        tfidf_similarity_dict = cosine_similarity_calculator.get_tfidf_similarity_dict(query_sentence_id)
        # print('[tfidf_similarity_dict]', tfidf_similarity_dict)
        spacy_similarity_dict = spacy_increaser.get_spacy_similarity_dict(query_sentence_id)
        # print('[spacy_similarity_dict]', spacy_similarity_dict)
        for sentence_id in tfidf_similarity_dict:
            # new_similarity_dict[sentence_id] = (tfidf_similarity_dict[sentence_id] + spacy_similarity_dict[sentence_id]) / 2
            new_similarity_dict[sentence_id] = (tfidf_similarity_dict[sentence_id]*2 + spacy_similarity_dict[sentence_id]) / 3
        return new_similarity_dict
    
    def _show_average_similarity_dict(self, nlu_dict, cosine_similarity_calculator, spacy_increaser):
        tfidf_sentence_average_similarity_dict = defaultdict(lambda: 0)
        for query_sentence_id in nlu_dict:
            tfidf_similarity_dict = cosine_similarity_calculator.get_tfidf_similarity_dict(query_sentence_id)
            for sentence_id in tfidf_similarity_dict:
                tfidf_sentence_average_similarity_dict[sentence_id] += tfidf_similarity_dict[sentence_id]
        for sentence_id in tfidf_sentence_average_similarity_dict:
            tfidf_sentence_average_similarity_dict[sentence_id] /= len(tfidf_sentence_average_similarity_dict)
        print('[tfidf_sentence_average_similarity_dict]', tfidf_sentence_average_similarity_dict)
        
        spacy_sentence_average_similarity_dict = defaultdict(lambda: 0)
        for query_sentence_id in nlu_dict:
            spacy_similarity_dict = spacy_increaser.get_spacy_similarity_dict(query_sentence_id)
            for sentence_id in spacy_similarity_dict:
                spacy_sentence_average_similarity_dict[sentence_id] += spacy_similarity_dict[sentence_id]
        for sentence_id in spacy_sentence_average_similarity_dict:
            spacy_sentence_average_similarity_dict[sentence_id] /= len(spacy_sentence_average_similarity_dict)
        print('[spacy_sentence_average_similarity_dict]', spacy_sentence_average_similarity_dict)
        
        tfidf_intent_average_similarity_dict = defaultdict(lambda: 0)
        len_dict = defaultdict(lambda: 0)
        for sentence_id in nlu_dict:
            intent_id = sentence_id.split('-')[0]
            tfidf_intent_average_similarity_dict[intent_id] += tfidf_sentence_average_similarity_dict[sentence_id]
            len_dict[intent_id] += 1 
        for intent_id in tfidf_intent_average_similarity_dict:
            tfidf_intent_average_similarity_dict[intent_id] /= len_dict[intent_id]
        print('[tfidf_intent_average_similarity_dict]', tfidf_intent_average_similarity_dict)
        
        spacy_intent_average_similarity_dict = defaultdict(lambda: 0)
        len_dict = defaultdict(lambda: 0)
        for sentence_id in nlu_dict:
            intent_id = sentence_id.split('-')[0]
            spacy_intent_average_similarity_dict[intent_id] += spacy_sentence_average_similarity_dict[sentence_id]
            len_dict[intent_id] += 1 
        for intent_id in spacy_intent_average_similarity_dict:
            spacy_intent_average_similarity_dict[intent_id] /= len_dict[intent_id]
        print('[spacy_intent_average_similarity_dict]', spacy_intent_average_similarity_dict)