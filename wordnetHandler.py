import statistics
from itertools import product
import itertools
import json
import yaml
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')  # for tokenizing


def wordnet_handler(data):
    data = yaml.load(data, Loader=yaml.FullLoader)
    for nlu in data['nlu']:
        if (nlu['intent'] == 'inform'): # intent 若為 inform 不需要處裡
            continue
        else:
            add_sentences(nlu)
    return data
    # return yaml.dump(data, default_flow_style=False)


# 增加每個意圖的訓練語句
def add_sentences(nlu):
    examples = list() # 先暫存到這裡，之後再更新到原始資料中
    for sentence in nlu['examples']:
        synonyms_sentence = list() # 由一個句子每個 token 的同義詞集合，組成的列表
        for token in nltk.wordpunct_tokenize(sentence):
            synsets = wn.synsets(token, pos=(wn.NOUN, wn.VERB))

            synonyms = set() # 每個 token 的同義詞集合
            synonyms.add(token) # 同義詞集合加入原始的 token
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
            examples.append(' '.join(word for word in sentence_word_list))
        
        # print(examples)
        # print()

    nlu['examples'] = examples