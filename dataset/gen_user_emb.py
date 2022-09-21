import json
import chardet
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter


def update_dict(dict, key, value=None):
    if key not in dict:
        if value is None:
            dict[key] = len(dict) + 1
        else:
            dict[key] = value


with open('./sci_word_dict.json', 'r', encoding='utf-8') as f:
    word_dict = json.load(f)
    glove_word_embedding = np.zeros(shape=(len(word_dict) + 1, 300))

    g = open('./glove.840B.300d.txt', 'rb')
    while True:
        line = g.readline()
        if len(line) == 0:
            break
        line = line.split()
        word = line[0].decode()
        if word in word_dict:
            index = word_dict[word]
            tp = [float(x) for x in line[1:]]
            glove_word_embedding[index] = np.array(tp)


user_index = {}
h = open('./Scientific.user2index', 'r')
for line in h.readlines():
    line = line.split('\t')
    user_id = line[0]
    user_idx = line[1].strip('\n')
    update_dict(user_index, user_id, user_idx)


user_word_cnt = Counter()

l = open('./sci_review.txt',encoding='utf-8')
user_embedding = np.zeros(shape=(len(user_index)+5,300))
for line in l.readlines():
    line_list = line.split('\t')
    user_id = line_list[0]
    if user_id not in user_index.keys():
        continue
    item_id = line_list[1]
    review_text = line_list[2].strip('\n')
    review_text_list = word_tokenize(review_text)
    user_idx = user_index[user_id]
    user_idx = int(user_idx)
    for word in review_text_list:
        if word in word_dict.keys():
            user_embedding[user_idx] = user_embedding[user_idx] + glove_word_embedding[word_dict[word]]
            user_word_cnt.update([user_idx])


for user_idx in range(len(user_index)):
    if user_word_cnt[user_idx]>0:
        user_embedding[user_idx] = user_embedding[user_idx]/user_word_cnt[user_idx]


np.save('user_emb.npy', user_embedding)
