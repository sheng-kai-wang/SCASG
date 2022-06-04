import numpy as np


class CosineSimilarityCalculator:
    def __init__(self, token_dict, nlu_dict) -> None:
        self.token_dict = token_dict
        self.nlu_dict = nlu_dict

    # 定義如何取得文件權重的 dict
    def _get_sentence_weight_dict(self, query_sentence) -> dict:
        sentence_weight_dict = dict()
        for token in self.token_dict:
            sentence_weight_dict[token] = 0
            # 跳過 statistics_node
            for document_node in self.token_dict[token][1:]:
                if query_sentence == document_node['id']:
                    sentence_weight_dict[token] = document_node['weight']
        return sentence_weight_dict

    # 定義如何取得文件向量
    def _get_sentence_vector(self, sentence_weight_dict) -> list:
        sentence_vector = sentence_weight_dict.values()
        return sentence_vector

    # 定義如何取得文件相似度
    def _get_sentence_similarity(self, sentence_a, sentence_b) -> float:
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

    # 找出 coupling 過高的語句，加到 coupling_too_high_dict 裡面
    def get_tfidf_similarity_dict(self, query_sentence_id) -> dict:
        similarity_dict = dict()
        query_sentence_vector = self._get_sentence_vector(self._get_sentence_weight_dict(query_sentence_id))
        # 取得 similarity_dict
        for sentence_id in self.nlu_dict:
            corpus_sentence_vector = self._get_sentence_vector(self._get_sentence_weight_dict(sentence_id))
            similarity_dict[sentence_id] = self._get_sentence_similarity(query_sentence_vector, corpus_sentence_vector)
        return similarity_dict
