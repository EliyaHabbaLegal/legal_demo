import json

MAX_EMBED_SIZE = 480


def load_jsonl_data(jsonl_input_file):
    examples = []
    with open(jsonl_input_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            cur_json = json.loads(line)
            examples.append(cur_json)
    return examples


def is_meaningful_text(text: str) -> bool:
    """
    This function returns whether a given string is likely to be meaningful.
    To be a meaningful text, the must hold the following properties:
    1. Contain at least a letter.
    2. Contain at least 5 words.
    3. Its length must be at least 10 characters.
    :param text: A given text.
    :return: True iff the given text is considered as meaningful
    """
    return text and any(c.isalpha() for c in text) and len(text.split()) > 4 and len(text) > 9


def split_text_by_tokenizer(text: str, tokenizer):
    """
    This function splits a text to a jsonl file according to the tokenizer's max embedding size.

    :param text: A given text to write to the outputfile
    :param tokenizer: A given tokenizer
    :return: A list of texts
    """
    period_token_id = tokenizer.convert_tokens_to_ids(".")
    semicolon_token_id = tokenizer.convert_tokens_to_ids(";")
    texts = list()
    while is_meaningful_text(text):
        tokens = tokenizer(text)['input_ids']

        if len(tokens) <= MAX_EMBED_SIZE:  # If whole text fits in the tokenizer
            texts.append(text)
            return texts

        # The text does not fit in the tokenizer - split by the last sentence before reaching max length
        if period_token_id in tokens[MAX_EMBED_SIZE::-1]:
            last_period_token_index = MAX_EMBED_SIZE \
                                      - tokens[MAX_EMBED_SIZE::-1].index(period_token_id) + 1
        elif semicolon_token_id in tokens[MAX_EMBED_SIZE::-1]:
            last_period_token_index = MAX_EMBED_SIZE - \
                                      tokens[MAX_EMBED_SIZE::-1].index(semicolon_token_id) + 1
        else:
            return texts

        texts.append(tokenizer.decode(tokens[1:last_period_token_index]))
        text = tokenizer.decode(tokens[last_period_token_index:-1]).strip()
    return texts


def small_get_data_points_from_paragraphs(paragraphs, tokenizer):
    """
    This function generates data points from given paragraphs.

    :param paragraphs: A given list of objects with 'text' field containing string representations of the paragraphs
    :param text_dict: A dictionary with the text information
    :param tokenizer:  A given tokenizer
    :return:
    """
    text = ''
    texts_results = []
    for p in paragraphs:
        if p.text.endswith(':'):  # If text ends with ':' -> the text continues in the next p
            text = p.text + ' '
        else:
            text_stripped = (text + p.text).replace('\n', ' ').strip()
            texts = split_text_by_tokenizer(text_stripped, tokenizer)
            texts_results.extend(texts)
            text = ''
    return texts_results


def get_data_points_from_paragraphs(paragraphs, tokenizer):
    text = ''
    i = 0
    texts_results = []
    while i < len(paragraphs):
        text_stripped = text.replace('\n', ' ').strip()
        while i < len(paragraphs) and (paragraphs[i].text.endswith(':') or \
                                       len(tokenizer((paragraphs[i].text).replace('\n', ' ').strip())['input_ids']) + \
                                       len(tokenizer(text_stripped)['input_ids']) <= MAX_EMBED_SIZE):
            text = paragraphs[i].text + ' '
            text_stripped += '\n' + text.replace('\n', ' ').strip()
            i += 1
        # text_stripped = text.replace('\n', ' ').strip()
        texts = split_text_by_tokenizer(text_stripped, tokenizer)
        texts_results.extend(texts)
        i += 1
    return texts_results

