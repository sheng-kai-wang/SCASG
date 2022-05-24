class SpacyIncreaser:
    def __init__(self, nlu_dict, nlp) -> None:
        self.nlu_dict = nlu_dict
        self.nlp = nlp

    def get_spacy_similarity_dict(self, query_sentence_id) -> dict:
        similarity_dict = dict()
        query_sentence = self.nlu_dict[query_sentence_id]
        for sentence_id in self.nlu_dict:
            sentence = self.nlu_dict[sentence_id]
            similarity_dict[sentence_id] = self.nlp(query_sentence).similarity(self.nlp(sentence))
        return similarity_dict
