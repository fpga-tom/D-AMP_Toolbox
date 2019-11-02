import numpy as np

SEQ_LEN_H = 45
SEQ_LEN_W = 45
channel_img = 8
im = [32000, 200, 1]


with open('/tomas/dict.txt') as f:
    tokens = []
    for l in f:
        tokens.append(l.lower())

    token_to_int = dict((t.strip(), i) for i, t in enumerate(tokens))

    print('total tokens: ' + str(len(token_to_int)))
    print(token_to_int)


def clip(x):
	return -1 if x == 0 else 1

lineno = 0
with open('/tomas/tokens.txt') as f:
    for jj in im:
        img = np.zeros([jj, SEQ_LEN_H, SEQ_LEN_W, channel_img])
        for i in range(jj):
            data_x = np.zeros([SEQ_LEN_H, SEQ_LEN_W, channel_img])
            for k in range(SEQ_LEN_H):
                for l in range(0,SEQ_LEN_W,1):
                    token = f.readline().lower().strip()
		    lineno += 1
                    ti = token_to_int[token]
                    data_x[k,l,0] = clip(ti % 2)
                    data_x[k,l,1] = clip((ti // 2) % 2)
                    data_x[k,l,2] = clip((ti // 4) % 2)
                    data_x[k,l,3] = clip((ti // 8) % 2)
                    data_x[k,l,4] = clip((ti // 16) % 2)
                    data_x[k,l,5] = clip((ti // 32) % 2)
                    data_x[k,l,6] = clip((ti // 64) % 2)
                    data_x[k,l,7] = clip((ti // 128) % 2)

#                    data_x[k,l,0] = ti % 2
#                    data_x[k,l,1] = (ti // 2) % 2
#                    data_x[k,l,2] = (ti // 4) % 2
#                    data_x[k,l+1,0] = (ti // 8) % 2
#                    data_x[k,l+1,1] = (ti // 16) % 2
#                    data_x[k,l+1,2] = (ti // 32) % 2
#                    data_x[k,l+2,0] = (ti // 64) % 2
#                    data_x[k,l+2,1] = (ti // 128) % 2
#                    data_x[k,l+2,2] = (ti // 256) % 2

            img[i,:,:,:] = data_x
        np.save('images' + str(jj) + '.npy', img.astype('float32'))


