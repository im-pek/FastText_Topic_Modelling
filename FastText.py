import numpy as np
import pandas as pd
from helper import *
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords 
import string

stopwords = set(stopwords.words('english'))
punctuation = set(string.punctuation) 

def cleaning(article):
    one = " ".join([i for i in article.lower().split() if i not in stopwords])
    two = "".join(i for i in one if i not in punctuation)
    return two

df = pd.read_csv(open("abstracts for 'automat'.csv", errors='ignore'))
df=df.astype(str)

text = df.applymap(cleaning)['paperAbstract']
text_list = [i.split() for i in text]

all_joined=[]

for element in text_list:
    joined=' '.join(element)
    all_joined.append(joined)
        
tagss=np.arange(len(text_list))

import multiprocessing ########
cores = multiprocessing.cpu_count() ########

entire_corpus=' '.join(all_joined)
entire_corpus_strings=entire_corpus.split(' ')

#model.build_vocab(entire_corpus_strings, update=True) #!
#model.train(entire_corpus_strings, total_examples=model.corpus_count, epochs=model.iter) #!

from gensim.models.fasttext import FastText

import multiprocessing ########
cores = multiprocessing.cpu_count() ########

model = FastText([entire_corpus_strings], min_count=1, iter=3, min_n = 2, max_n=25, word_ngrams = 1, sg=1, hs=1, negative=2) #!workers=cores, max_n = 20, size=4, window=3
#min_count: ignores all words with total frequency lower than this
#sg=0 is CBOW, sg=1 is Skip-Gram
#hs=1 employs hierarchical softmax
#negative > 0 employs negative sampling. 2-5 for large datasets, 5-20 for small datasets

#model.n_similarity(words, blah) #similarity between (all) words in two lists

top_wordsss=model.wv.most_similar(positive=entire_corpus, topn=30)#, negative=['good']))#, topn=20)) #!

top_MI_values_list=[]
for tuplee in top_wordsss:
    listed=list(tuplee)
    top_MI_values_list.append(listed)
top_wordies=[]
for ix,thingy in enumerate(top_MI_values_list):
    top_wordies.append(top_MI_values_list[ix][0])
#print (top_wordsss) #shows MI values too
print ('Fasttext Topics:')
print (top_wordies)
print ('\n')