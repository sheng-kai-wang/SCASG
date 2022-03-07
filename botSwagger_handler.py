import json
from jsonpath_ng import parse
import spacy


class botSwaggerHandler:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_lg")  # for tokenizing
        self.cohesion_too_low_list = list()
        self.coupling_too_high_list = list()

    def check_sentences(self, data) -> json:
        data = json.loads(data)
        examples = list()  # 裝所有 intent 訓練語句用的 list
        # jsonpath 找出 botSwagger 當中的訓練語句
        for context in parse('$..["x-input-template"]..*').find(data):
            # 以下先斷詞，後刪去 stopwords
            sentences = list()  # 裝某個 intent 訓練語句用的 list
            for sentence in context.value:
                filtered_sentence = list()  # 裝刪去 stopwords 後的 token 的 list
                for word in self.nlp(sentence):  # 斷詞出 token
                    if word.is_stop == False:
                        filtered_sentence.append(word)
                # token 的 list 重組成句子
                sentences.append(
                    ' '.join(token.text for token in filtered_sentence))
            examples.append(sentences)  # 加入各個 intent 的句子
            # examples.append(context.value)

        # print('examples', examples)
        self.check_intent_cohesion(examples)
        # self.check_intent_coupling(examples)

        return 'botSwagger\'s intent check is finished'

    # 挑出同一 intent 裡 cohesion 較低的句子
    def check_intent_cohesion(self, examples):
        for sentences in examples:

            # sentence_list = list()
            # for sentence in sentences:
            #     filtered_sentence = list()
            #     for word in self.nlp(sentence):
            #         if word.is_stop == False:
            #             filtered_sentence.append(word)
            #     sentence_list.append(' '.join(token.text for token in filtered_sentence))
            # print('sentence_list', sentence_list)

            base_sentence = sentences[0]  # 用每個 intent 的第一句當作比較的標準
            similarity_list = list()  # 裝相似度數值的 list
            for sentence in sentences:
                # print(sentence)
                if (self.nlp(base_sentence).similarity(self.nlp(sentence)) < 0.6):  # 比較相似度
                    self.cohesion_too_low_list.append(sentence)
                similarity_list.append(
                    self.nlp(base_sentence).similarity(self.nlp(sentence)))
            print(similarity_list)
            # print()
        print('cohesion_too_low_list', self.cohesion_too_low_list)

    # 挑出不同 intent 裡 coupling 較高的句子
    def check_intent_coupling(self, examples):
        # 將各個 intent 的所有句子 全部串接成一句 總共十句 兩兩互相比較
        compare_list = list() # 裝上述那些句子的 list
        for sentences in examples:
            spliced_sentence = ''
            for sentence in sentences:
                spliced_sentence += sentence # 串接成一句
            compare_list.append(spliced_sentence)

        print('compare_list', compare_list)

        similarity_list = list()
        for base_compare in compare_list:
            for to_be_compared in compare_list:
                similarity_list.append(
                    self.nlp(base_compare).similarity(self.nlp(to_be_compared))) # 兩兩互相比較相似度
        print('similarity_list', similarity_list)
