import json
import chardet
from nltk.tokenize import word_tokenize


sci_review=open('sci_review.txt','w')

with open('./Industrial_and_Scientific.json',encoding='utf-8') as f:
    for line in f.readlines():
        line=json.loads(line)
        try:
            if 'reviewText' in line and line['reviewText'].strip()!='':
                review_text = line['reviewText'].strip().replace('\n',' ')
                review_text = review_text.lower()
            if 'reviewerID' in line and line['reviewerID'].strip()!='':
                reviewer_id = line['reviewerID'].strip()
            if 'asin' in line and line['asin'].strip()!='':
                item_id = line['asin'].strip()
            new_line = reviewer_id+'\t'+item_id+'\t'+review_text+'\n'
            sci_review.write(new_line)
        except Exception as e:
            print(line)
            print(e)
    f.close()
    sci_review.close()
