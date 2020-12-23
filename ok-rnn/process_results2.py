import my_txtutils
import tensorflow.compat.v1 as tf
import numpy as np
import math
SEQLEN = 30
BATCHSIZE = 200
ALPHASIZE = my_txtutils.ALPHASIZE
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 0.8    # some dropout
tf.disable_v2_behavior()

import kenlm



with open('txts/gal_eph_new.txt', encoding='utf-16') as f:
  s=f.read()

author='checkpoints/rnn_train_1608440693-210000000'

def get_scores(corpora, use_log=False):
  probs=[]
  with tf.Session() as sess:
      # new_saver = tf.train.import_meta_graph('checkpoints/rnn_train_1512567262-0.meta')
      new_saver = tf.train.import_meta_graph(author+'.meta')
      new_saver.restore(sess, author)
      x = my_txtutils.convert_from_alphabet(ord(corpora[0]))
      x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

      # initial values
      y = x
      h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
      for i in range(1,len(corpora)):
          yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})
          #!nvidia-smi
          next_ord =  my_txtutils.convert_from_alphabet(ord(corpora[i]))
          # If sampling is bedone from the topn most likely characters, the generated text
          # is more credible and more "english". If topn is not set, it defaults to the full
          # distribution (ALPHASIZE)
          #print(yo.shape)
          if use_log:
            probs.append(math.log(yo[0][next_ord]))
          else:
            probs.append(yo[0][next_ord])
          # Recommended: topn = 10 for intermediate checkpoints, topn=2 or 3 for fully trained checkpoints

          # c = my_txtutils.sample_from_probabilities(yo, topn=2)
          y = np.array([[next_ord]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
          # c = chr(my_txtutils.convert_to_alphabet(c))
      return probs

def cppx(scores, corpus):
  nw = len(corpus.split())
  nc = len(corpus)
  c = math.exp(sum(scores) / nc)
  print(c, nw, nc)
  return 2 ** (c * nc / nw)

def cppx2(scores, corpus):
  nw = len(corpus.split())
  nc = len(corpus)
  ip = np.sum(-np.log(scores)) / nc
  c = math.exp(ip)
  print(ip, nw, nc)
  return 2 ** (ip * nc / nw)

scores = get_scores(s)
cppx2(scores, s)
nc = len(s)
nw = len(s.split())
ip2 = math.exp(sum(scores) / nc)
print(ip2, nw, nc)


model = kenlm.Model('txteval/text.arpa')

sc = [score for score,_,_ in model.full_scores(s)]
ip = math.exp(sum(sc)/nw)

