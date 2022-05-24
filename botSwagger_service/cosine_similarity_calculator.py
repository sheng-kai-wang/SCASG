import numpy as np
from collections import defaultdict


class CosineSimilarityCalculator:
    def __init__(self, token_dict) -> None:
        self.token_dict = token_dict

    # 定義如何取得文件權重的 dict
    def __get_sentence_weight_dict(self, query_sentence) -> dict:
        sentence_weight_dict = dict()
        for token in self.token_dict:
            sentence_weight_dict[token] = 0
            # 跳過 statistics_node
            for document_node in self.token_dict[token][1:]:
                if query_sentence == document_node['id']:
                    sentence_weight_dict[token] = document_node['weight']
        return sentence_weight_dict

    # 定義如何取得文件向量
    def __get_sentence_vector(self, sentence_weight_dict):
        sentence_vector = sentence_weight_dict.values()
        return sentence_vector

    # 定義如何取得文件相似度
    def __get_sentence_similarity(self, sentence_a, sentence_b):
        # Dot and norm
        dot = sum(a*b for a, b in zip(sentence_a, sentence_b))
        norm_a = sum(a*a for a in sentence_a) ** 0.5
        norm_b = sum(b*b for b in sentence_b) ** 0.5
        # 避免分母為 0
        if norm_a * norm_b == 0.0:
            return 0.0
        # Cosine similarity
        cos_similarity = dot / (norm_a*norm_b)
        # 避免 NaN
        if np.isnan(cos_similarity):
            cos_similarity = 0.0
        return cos_similarity

    # 找出 coupling 過高的語句，加到 coupling_too_high_dict 裡面 (有 bug) (需要重構)
    def get_similarity_dict(self, nlu_dict, query_sentence_id):
        similarity_dict = dict()
        query_sentence_vector = self.__get_sentence_vector(self.__get_sentence_weight_dict(query_sentence_id))
        # 取得 similarity_dict
        for sentence_id in nlu_dict:
            corpus_sentence_vector = self.__get_sentence_vector(self.__get_sentence_weight_dict(sentence_id))
            similarity_dict[sentence_id] = self.__get_sentence_similarity(query_sentence_vector, corpus_sentence_vector)
        return similarity_dict
        # 將相同 intent 的語句的相似度相加
        # intent_sum_dict = defaultdict(lambda: 0.0)
        # for sentence_id in similarity_dict:
        #     intent_id = sentence_id.split('-')[0]
        #     intent_sum_dict[intent_id] += similarity_dict[sentence_id]
        # 如果有 intent 的相似度總和，高於查詢 intent 的相似度總和，則加到 coupling_too_high_dict 裡面
        # query_intent_id = query_sentence_id.split('-')[0]
        # query_intent_sum = intent_sum_dict[query_intent_id]
        # for intent_id in intent_sum_dict:
        #     if intent_sum_dict[intent_id] > query_intent_sum:
        #         print(intent_id, intent_sum_dict[intent_id])
        # for intent_id in intent_sum_dict:
        #     if intent_sum_dict[intent_id] > intent_sum_dict[query_sentence_id.split('-')[0]]:
        #         # 找出 intent 中，是哪一句過高
        #         max = 0.0
        #         for sentence_id in similarity_dict:
        #             if sentence_id.split('-')[0] == intent_id:
        #                 if similarity_dict[sentence_id] > max:
        #                     max = similarity_dict[sentence_id]
        #         for sentence_id in similarity_dict:
        #             if similarity_dict[sentence_id] == max:
        #                 coupling_too_high_dict[query_sentence_id].append(sentence_id)
