import json
from collections import OrderedDict
import oyaml as yaml  # for ordered yaml file
import statistics
from itertools import product
from xmlrpc.client import boolean

from nltk.corpus import wordnet as wn
import nltk
from nltk.tokenize import SpaceTokenizer

import spacy


class SentencesHandler:
    def __init__(self) -> None:
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')  # for tokenizing
        nltk.download('punkt') # for tokenizing
        nltk.download('omw-1.4')
        self.nlp = spacy.load("en_core_web_lg")

    # 取得生成的 nlu

    def get_nlu(self, data) -> json:
        data = yaml.load(data, Loader=yaml.FullLoader)
        new_nlus = list()
        for nlu in data['nlu']:
            # intent 若為 inform 或 auto_get_location 不需要處裡
            if (nlu['intent'] == 'inform' or nlu['intent'] == 'auto_get_location'):
            # if (nlu['intent'] == 'inform'):
                new_nlus.append(nlu)
                continue
            else:
                new_nlus.append(self.add_by_wordnet(nlu))

        data['nlu'] = new_nlus
        return json.dumps(data, sort_keys=False)

    # 增加每個意圖的訓練語句

    def add_by_wordnet(self, nlu) -> OrderedDict:
        new_nlu = OrderedDict()
        examples = list()  # 先暫存到這裡，之後再更新到原始資料中

        for sentence in nlu['examples'].split("\n"):
            synonyms_sentence = list()  # 由一個句子每個 token 的同義詞集合，組成的列表
            
            # 也許可以這樣斷詞
            # tk = SpaceTokenizer()
            # tk.tokenize(sentence)
            
            for token in nltk.word_tokenize(sentence):
                synonyms = set()  # 每個 token 的同義詞集合
                synonyms.add(token)  # 同義詞集合加入原始的 token

                if (token == 'I'):  # I 個別處理
                    synonyms.add('i')
                elif (token.isdigit()): # 數字不處理
                    pass
                elif (token.is_stop == False): # stopwords 跳過不處理
                    pass
                else:
                    synsets = wn.synsets(token, pos=(wn.NOUN, wn.VERB))

                    for synset in synsets:
                        for synonym in synset.lemma_names():
                            allsyns1 = set(ss for ss in wn.synsets(token))
                            allsyns2 = set(ss for ss in wn.synsets(synonym))
                            # 評估新產生的同義詞與原始的 token 的相似度，利用笛卡爾乘積交叉比較兩組同義詞的相似度，並計算平均值
                            mean = statistics.mean(
                                (wn.wup_similarity(s1, s2) or 0) for s1, s2 in product(allsyns1, allsyns2))
                            if mean > 0.36:
                            # if (True):
                                synonyms.add(synonym)

                synonyms_sentence.append(synonyms)

            # 同義詞集合組成的列表，交叉組合成新的句子
            for sentence_word_list in list(product(*synonyms_sentence)):
                # new_sentence = ''
                # for word in sentence_word_list:   
                    # if (word in ('(', ')', '[', ']')):
                    #     new_sentence += word
                    # elif (word == ','):
                    #     new_sentence += word + ' '
                    # else:
                    #     new_sentence += word + ' '
                
                new_sentence = ' '.join(word for word in sentence_word_list)
                if (self.remove_by_spacy(sentence, new_sentence)):
                # if (True):
                    examples.append(new_sentence)

        # nlu['examples'] = examples
        new_nlu['intent'] = nlu['intent']
        new_nlu['examples'] = '\n'.join(examples)
        return new_nlu

    # 移除掉與原句差異過大的新句子

    def remove_by_spacy(self, pre_sentence, new_sentence) -> boolean:
        score = self.nlp(pre_sentence).similarity(self.nlp(new_sentence))
        return score > 0.9
