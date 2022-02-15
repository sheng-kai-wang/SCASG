import yaml
import statistics
from itertools import product
import itertools
from xmlrpc.client import boolean
from sqlalchemy import true

from nltk.corpus import wordnet as wn
import nltk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')  # for tokenizing

import spacy

nlp = spacy.load("en_core_web_lg")


def sentences_handle(data):
    data = yaml.load(data, Loader=yaml.FullLoader)
    for nlu in data['nlu']:
        # intent 若為 inform 或 auto_get_location 不需要處裡
        if (nlu['intent'] == 'inform' or nlu['intent'] == 'auto_get_location'): 
            continue
        else:
            add_by_wordnet(nlu)
    return yaml.dump(data, default_flow_style=False)


# 增加每個意圖的訓練語句
def add_by_wordnet(nlu):
    examples = list() # 先暫存到這裡，之後再更新到原始資料中
    for sentence in nlu['examples']:
        synonyms_sentence = list() # 由一個句子每個 token 的同義詞集合，組成的列表
        for token in nltk.wordpunct_tokenize(sentence):
            synonyms = set() # 每個 token 的同義詞集合
            synonyms.add(token) # 同義詞集合加入原始的 token

            if (token == 'I'): # I 個別處理
                synonyms.add('i')
            else:
                synsets = wn.synsets(token, pos=(wn.NOUN, wn.VERB))

                for synset in synsets:
                    for synonym in synset.lemma_names():
                        allsyns1 = set(ss for ss in wn.synsets(token))
                        allsyns2 = set(ss for ss in wn.synsets(synonym))
                        # 評估新產生的同義詞與原始的 token 的相似度，利用笛卡爾乘積交叉比較兩組同義詞的相似度，並計算平均值
                        mean = statistics.mean((wn.wup_similarity(s1, s2) or 0) for s1, s2 in product(allsyns1, allsyns2))
                        if mean > 0.32:
                            synonyms.add(synonym)
                
            synonyms_sentence.append(synonyms)
        
        # 同義詞集合組成的列表，交叉組合成新的句子
        for sentence_word_list in list(itertools.product(*synonyms_sentence)):
            new_sentence = ' '.join(word for word in sentence_word_list)
            if (remove_by_spacy(sentence, new_sentence)):
                examples.append(new_sentence)

    nlu['examples'] = examples


# 移除掉與原句差異過大的新句子
def remove_by_spacy(pre_sentence, new_sentence) -> boolean:
    score = nlp(pre_sentence).similarity(nlp(new_sentence))
    if (score > 0.9):
        return true