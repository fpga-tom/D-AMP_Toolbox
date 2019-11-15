import numpy as np
import xxhash
import pickle

SEQ_LEN_H = 26
SEQ_LEN_W = 4
channel_img = 40
im = [160000, 16000]

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
model = gensim.models.Word2Vec(sentences=sentences, size=channel_img, workers=10, min_count=1)
model.init_sims()


#with open('/tomas/dict.txt') as f:
#    tokens = []
#    for l in f:
#        tokens.append(l.lower())
#
#    token_to_int = dict((t.strip(), i) for i, t in enumerate(tokens))
#
#    print('total tokens: ' + str(len(token_to_int)))
#    print(token_to_int)


def clip(x):
	return (-1 if x == 0 else 1)

lineno = 0
int_to_token = dict()
with open('/tomas/tokens.txt') as f:
    for jj in im:
        img = np.zeros([jj, SEQ_LEN_H * SEQ_LEN_W, channel_img])
        for i in range(jj):
            data_x = np.zeros([SEQ_LEN_H * SEQ_LEN_W,  channel_img])
            for k in range(SEQ_LEN_H * SEQ_LEN_W):
                    token = f.readline().strip()
		    lineno += 1
#		    x = xxhash.xxh32()
#		    x.update(token)
#		    ti = x.intdigest()
#		    int_to_token[ti] = token
#		    data_x[k,l,:] = [clip((ti // 2**ci) % 2) for ci in range(channel_img)]
		    try:
			    v = model.wv.word_vec(token, use_norm=True)
			    data_x[k,:] = v
		    except Exception:
			pass

            img[i,:,:] = data_x
        np.save('images' + str(jj) + '.npy', img.astype('float32'))

with open('/tomas/test_tokens.txt') as f:
	jj = 1
        img = np.zeros([jj, SEQ_LEN_H * SEQ_LEN_W, channel_img])
        for i in range(jj):
            data_x = np.zeros([SEQ_LEN_H * SEQ_LEN_W,  channel_img])
            for k in range(SEQ_LEN_H * SEQ_LEN_W):
                    token = f.readline().strip()
		    lineno += 1
		    #x = xxhash.xxh32()
		    #x.update(token)
		    #ti = x.intdigest()
		    #int_to_token[ti] = token
		    #data_x[k,l,:] = [clip((ti // 2**ci) % 2) for ci in range(channel_img)]
		    try:
			    v = model.wv.word_vec(token, use_norm=True)
			    data_x[k,:] = v
		    except Exception:
			pass

            img[i,:,:] = data_x
        np.save('images' + str(jj) + '.npy', img.astype('float32'))

model.save('saved_models/LDAMP/w2v.model')


with open('int_to_token.p', 'wb') as fp:
    pickle.dump(int_to_token, fp, protocol=pickle.HIGHEST_PROTOCOL)


