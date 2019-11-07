from gensim.test.utils import datapath
from gensim import utils

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('/tomas/tokens.txt')
	doc = []
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
	    doc.append(line.strip())
	    if line == '\n':
		yield doc
		doc = []
	

import gensim.models

sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences, size=32, workers=10)


print(model.wv['sizeof'])
print(model.wv.similarity('sizeof', 'int'))
print(model.wv.similarity('sizeof', 'main'))
print(model.wv.most_similar('printk'))
print(model.wv.similar_by_vector(model.wv['printk'], topn=1))
