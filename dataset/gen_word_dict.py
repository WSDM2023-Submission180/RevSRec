import json
import chardet
from nltk.tokenize import word_tokenize
from collections import Counter
import json


reviews = {}
word_dict = {}
word_cnt = Counter()

def update_dict(dict, key, value=None):
    if key not in dict:
        if value is None:
            dict[key] = len(dict) + 1
        else:
            if key in dict.keys():
                dict[key] = dict[key] + value
            else:
                dict[key] = value


with open('./sci_review.txt',encoding='utf-8') as f:
    for line in f.readlines():
        line_list = line.split('\t')
        user_id = line_list[0]
        item_id = line_list[1]
        review_text = line_list[2].strip('\n')
        review_text_list = word_tokenize(review_text)
        word_cnt.update(review_text_list)
        update_dict(reviews, user_id, review_text)

    word = [k for k, v in word_cnt.items() if v > 3]
    word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}

    jsonstr = json.dumps(word_dict)
    filename = open('sci_word_dict.json','w')
    filename.write(jsonstr)
