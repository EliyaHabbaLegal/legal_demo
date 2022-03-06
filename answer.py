class Answer:
    def __init__(self, tokenizer, input_ids, answer_start_ind, answer_end_index, answer_start_confidence, answer_end_confidence):
        self._tokenizer = tokenizer
        self._input_ids = input_ids
        self._answer_start_ind = answer_start_ind
        self._answer_end_index = answer_end_index
        self.answer_start_confidence = answer_start_confidence
        self.answer_end_confidence = answer_end_confidence
        self.significant_answer = False
    def convert_tokens_to_str(self, tokens):
        answer_tokens = self._tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=True)
        answer_tokens_to_string = self._tokenizer.convert_tokens_to_string(answer_tokens)
        return answer_tokens_to_string

    def get_string_from_token_indexes(self, start_index, end_index):
        tokens = self._input_ids[start_index:end_index]
        string_from_tokens = self.convert_tokens_to_str(tokens)
        return string_from_tokens

    def combine_answer_from_context(self):
        self.answer_itself = self.get_string_from_token_indexes(self._answer_start_ind, self._answer_end_index + 1)
        self.beginning_of_context_until_answer = self.get_string_from_token_indexes(0, self._answer_start_ind)
        self.end_of_context_from_the_answer = self.get_string_from_token_indexes(self._answer_end_index + 1, -1)
        if self.answer_itself:
            self.significant_answer = True
