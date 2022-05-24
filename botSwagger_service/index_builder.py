import math
from collections import defaultdict
# import statistics
# from itertools import product

# import WordNet
import nltk


class IndexBuilder:
    def __init__(self, nlu_dict, lemmatizer) -> None:
        self.lemmatizer = lemmatizer
        self.token_dict = self.__build_index(nlu_dict)
        self.token_dict = self.__calculate_weight(nlu_dict, self.token_dict)

    def get_token_dict(self) -> dict:
        return self.token_dict

    def __build_index(self, nlu_dict) -> dict:
        token_dict = defaultdict(list)
        stop_words = set(nltk.corpus.stopwords.words('english'))

        for sentence_id in nlu_dict:
            sentence = nlu_dict[sentence_id]
            tokens = nltk.word_tokenize(sentence.lower())
            # remove stop words
            tokens = [w for w in tokens if w not in stop_words]
            # lemmatization
            tokens = [self.lemmatizer.lemmatize(w) for w in tokens]

            for token in tokens:
                token = token.strip('\n').strip()

                if token not in token_dict:  # token不存在就創造list
                    token_dict[token] = list()
                    statistics_node = dict()  # list的第一個節點
                    statistics_node['df'] = 1
                    token_dict[token].append(statistics_node)

                    document_node = dict()  # list第二個以後的節點
                    document_node['id'] = sentence_id
                    document_node['tf'] = 1
                    document_node['weight'] = None  # 等全部跑完，再計算，因為df值會一直變動
                    token_dict[token].append(document_node)

                else:  # token存在
                    is_match = False
                    # 跳過 statistics_node
                    for document_node in token_dict[token][1:]:
                        # 已經存在這篇文章
                        if document_node['id'] == sentence_id:
                            is_match = True
                            document_node['tf'] += 1

                    if not is_match:  # token_dict有這個token，但沒有這篇文章
                        document_node = dict()
                        document_node['id'] = sentence_id
                        document_node['tf'] = 1
                        token_dict[token].append(document_node)

                    token_dict[token][0]['df'] = len(token_dict[token]) - 1  # 減去 statistic_node

        return token_dict

    def __calculate_weight(self, nlu_dict, token_dict) -> dict:
        for token in token_dict:
            for document_node in token_dict[token][1:]:
                # (1 + log10(tf)) * log10(N/df)
                document_node['weight'] = (1 + math.log10(document_node['tf'])) * math.log10(len(nlu_dict)/token_dict[token][0]['df'])
        return token_dict
