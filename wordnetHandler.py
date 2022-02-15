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
        add_sentences(nlu)
    return data


def add_sentences(nlu):
    examples = list()
    for sentence in nlu['examples']:
        synonyms_sentence = list()
        for token in nltk.wordpunct_tokenize(sentence):
            synsets = wn.synsets(token, pos=(wn.NOUN, wn.VERB))

            synonyms = set()
            synonyms.add(token)
            for synset in synsets:
                for synonym in synset.lemma_names():
                    allsyns1 = set(ss for ss in wn.synsets(token))
                    allsyns2 = set(ss for ss in wn.synsets(synonym))
                    mean = statistics.mean((wn.wup_similarity(s1, s2) or 0) for s1, s2 in product(allsyns1, allsyns2))
                    if mean > 0.32:
                        synonyms.add(synonym)
            
            synonyms_sentence.append(synonyms)
        
        for sentence_word_list in list(itertools.product(*synonyms_sentence)):
            examples.append(' '.join(word for word in sentence_word_list))
        
        # print(examples)
        # print()

    nlu['examples'] = examples