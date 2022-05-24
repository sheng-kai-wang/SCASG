from nltk.corpus import wordnet as wn


class WordnetIncreaser:
    def __init__(self, token_dict, lemmatizer, nlp) -> None:
        self.nlp = nlp
        self.token_dict = token_dict
        self.synonym_dict = self.__get_synonym_dict(lemmatizer)
        self.updated_token_dict = self.__update_index()

    def get_updated_token_dict(self) -> dict:
        return self.updated_token_dict

    def __get_synonym_dict(self, lemmatizer) -> dict:
        synonym_dict = dict()
        for token in self.token_dict:
            token_synonym_set = set()
            synsets = wn.synsets(token, pos=(wn.NOUN, wn.VERB))
            for synset in synsets:
                for synonym in synset.lemma_names():
                    token_nlp = self.nlp(token)
                    synonym_nlp = self.nlp(synonym)
                    if token_nlp.vector_norm and synonym_nlp.vector_norm:
                        score = token_nlp.similarity(synonym_nlp)
                        if score > 0.6:
                            # lemmatization and to lower case
                            synonym = lemmatizer.lemmatize(synonym.lower())
                            token_synonym_set.add(synonym)
                            token_synonym_set.discard(token)

            if len(token_synonym_set) != 0:
                synonym_dict[token] = token_synonym_set

        print('[synonym_dict]', synonym_dict)
        return synonym_dict

    def __update_index(self) -> dict:
        updated_token_dict = self.token_dict

        for token in self.synonym_dict:
            for synonym in self.synonym_dict[token]:

                # token不存在就用原始 list 複製一份給新的
                if synonym not in updated_token_dict:
                    updated_token_dict[synonym] = updated_token_dict[token]
                    for document_node in updated_token_dict[synonym][1:]:
                        # weight 重新計算
                        self.__update_document_node_weight(token, synonym, document_node)

                # token存在, 屬於原始的 list
                elif synonym in self.token_dict:
                    if len(updated_token_dict['*' + synonym]) == 0:  # 第一次出現
                        updated_token_dict['*' + synonym] = updated_token_dict[token]  # 擴充後的 token 跟原始的 token 重複了
                        for document_node in updated_token_dict['*' + synonym][1:]:
                            # weight 重新計算
                            self.__update_document_node_weight(token, synonym, document_node)

                    else:  # 不是第一次出現
                        new_list = updated_token_dict[token]
                        for document_node in new_list[1:]:  # 跳過 statistics_node
                            # weight 重新計算
                            self.__update_document_node_weight(token, synonym, document_node)
                        # 新舊 list 相加
                        synonym = '*' + synonym
                        self.__update_token_list(updated_token_dict, synonym, new_list)

                # token 存在，屬於新生成的 list
                else:
                    new_list = updated_token_dict[token]
                    for document_node in new_list[1:]:  # 跳過 statistics_node
                        # weight 重新計算
                        self.__update_document_node_weight(token, synonym, document_node)
                    # 新舊 list 相加
                    self.__update_token_list(updated_token_dict, synonym, new_list)

        print('[updated_token_dict]', updated_token_dict)
        return updated_token_dict

    def __update_document_node_weight(self, token, synonym, document_node) -> None:
        similarity = self.nlp(token).similarity(self.nlp(synonym))
        document_node['weight'] *= similarity

    def __update_token_list(self, updated_token_dict, synonym, new_list) -> None:
        for document_node in updated_token_dict[synonym][1:]:
            document_node_match = 0
            for new_document_node in new_list[1:]:
                if document_node == new_document_node:
                    document_node_match += 1
                    document_node['tf'] += new_document_node['tf']  # tf 相加
                    document_node['weight'] += new_document_node['weight']
            document_node['weight'] /= document_node_match + 1  # weight 取算術平均數
        updated_token_dict[synonym][0]['df'] = len(updated_token_dict[synonym]) - 1  # df 重新計算
