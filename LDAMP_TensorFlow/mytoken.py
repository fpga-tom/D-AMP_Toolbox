import numpy as np

SEQ_LEN_H = 128
SEQ_LEN_W = 80
channel_img = 3
im = [4000, 200, 1]


with open('/tmp/dict.txt') as f:
    tokens = []
    for l in f:
        tokens.append(l.lower())

    token_to_int = dict((t, i) for i, t in enumerate(tokens))

    print('total tokens: ' + len(token_to_int))


with open('/tmp/train.txt') as f:
    for jj in im:
        img = np.zeros([jj, SEQ_LEN_H, SEQ_LEN_W, channel_img])
        for i in range(jj):
            data_x = np.zeros([SEQ_LEN_H, SEQ_LEN_W, channel_img])
            for k in range(SEQ_LEN_H):
                for l in range(0,SEQ_LEN_W,3):
                    token = f.readline().lower()
                    ti = token_to_int[token]
                    data_x[k,l,0] = data_x % 2
                    data_x[k,l,1] = (data_x // 2) % 2
                    data_x[k,l,2] = (data_x // 4) % 2
                    data_x[k,l+1,0] = (data_x // 8) % 2
                    data_x[k,l+1,1] = (data_x // 16) % 2
                    data_x[k,l+1,2] = (data_x // 32) % 2
                    data_x[k,l+2,0] = (data_x // 64) % 2
                    data_x[k,l+2,1] = (data_x // 128) % 2
                    data_x[k,l+2,2] = (data_x // 256) % 2

            img[i,:,:,:] = data_x
        np.save('images' + str(jj) + '.npy', img.astype('float32'))


