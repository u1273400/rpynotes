import my_txtutils
import tensorflow.compat.v1 as tf
import numpy as np
import math
import torch.nn as nn
import torch

SEQLEN = 30
BATCHSIZE = 1
ALPHASIZE = my_txtutils.ALPHASIZE
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 0.8  # some dropout
tf.disable_v2_behavior()

import kenlm

with open('txts/gal_eph_new.txt', encoding='utf-16') as f:
    s = f.read()

author = 'checkpoints/rnn_train_1608440693-210000000'


def get_scores(corpora, use_log=False):
    probs = []
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(author + '.meta')
        new_saver.restore(sess, author)
        x = my_txtutils.convert_from_alphabet(ord(corpora[0]))
        x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

        # initial values
        y = x
        h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
        for i in range(1, len(corpora)):
            yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})
            next_ord = my_txtutils.convert_from_alphabet(ord(corpora[i]))
            # If sampling is bedone from the topn most likely characters, the generated text
            # is more credible and more "english". If topn is not set, it defaults to the full
            # distribution (ALPHASIZE)
            if use_log:
                probs.append(math.log(yo[0][next_ord]))
            else:
                probs.append(yo[0][next_ord])
            # Recommended: topn = 10 for intermediate checkpoints, topn=2 or 3 for fully trained checkpoints

            y = np.array([[next_ord]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        return probs

nc = len(s)
nw = len(s.split())

model = kenlm.Model('txteval/text.arpa')

sc = [score for score, _, _ in model.full_scores(s)]

gpufound = torch.cuda.is_available()
device = 'cuda' if gpufound else 'cpu'
print('using gpu' if gpufound else 'using cpu')

class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN2, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(BATCHSIZE, self.hidden_size, device=device)


class RNN1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN1, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.GRUCell(input_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input, hidden)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(BATCHSIZE, self.hidden_size, device=device)


def mb2t(rows):
    rows = rows.transpose()
    tensor = torch.zeros(rows.shape[0], rows.shape[1], ALPHASIZE, device=device)
    for i, row in enumerate(rows):
        for j, letter_code in enumerate(row):
            tensor[i][j][letter_code] = 1
    return tensor


def tf_play(size):
    ncnt = 0
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(author + '.meta')
        new_saver.restore(sess, author)
        x = my_txtutils.convert_from_alphabet(ord("L"))
        x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

        # initial values
        y = x
        h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
        for i in range(size):
            yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})

            # If sampling is bedone from the topn most likely characters, the generated text
            # is more credible and more "english". If topn is not set, it defaults to the full
            # distribution (ALPHASIZE)

            # Recommended: topn = 10 for intermediate checkpoints, topn=2 or 3 for fully trained checkpoints

            c = my_txtutils.sample_from_probabilities(yo, topn=2)
            y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
            c = chr(my_txtutils.convert_to_alphabet(c))
            print(c, end="")

            if c == '\n':
                ncnt = 0
            else:
                ncnt += 1
            if ncnt == 70:
                print("")
                ncnt = 0

def play(size, rnn, path):
    probs = []

    rnn.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    hidden = torch.zeros(1, INTERNALSIZE, device=device)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

    x = my_txtutils.convert_from_alphabet(ord("L"))
    x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    y = mb2t(x)
    ncnt = 0
    for i in range(size):
        yo, hidden = rnn(y[0], hidden)
        c = my_txtutils.sample_from_probabilities(yo.detach().numpy(), topn=2)
        y = mb2t(np.array([[c]]))  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        c = chr(my_txtutils.convert_to_alphabet(c))
        print(c, end="")

        if c == '\n':
            ncnt = 0
        else:
            ncnt += 1
        if ncnt == 70:
            print("")
            ncnt = 0

def pt_scores(corpora, rnn, path):
    probs = []

    rnn.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    hidden = torch.zeros(1, INTERNALSIZE, device=device)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]

    x = [my_txtutils.convert_from_alphabet(ord(c)) for c in corpora]
    x = np.array([x])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

    line_tensor = mb2t(x)

    for i in range(1, line_tensor.size()[0]):
        yo, hidden = rnn(line_tensor[i-1], hidden)
        next_ord = my_txtutils.convert_from_alphabet(ord(corpora[i]))
        probs.append(yo[0][next_ord])

    return torch.stack(probs).detach().numpy()


def wppx_logprob(scores, corpus):
    nw = len(corpus.split()) + 1
    ip = -np.sum(scores) / nw
    print(ip, 2**ip, math.exp(ip))
    return 10**ip

def wppx(sentence):
    """
    Compute perplexity of a sentence.
    @param sentence One full sentence to score.  Do not include <s> or </s>.
    """
    words = len(sentence.split()) + 1 # For </s>
    return 10.0**(-model.score(sentence) / words)


def ln2lt(lst):
    return np.log10(np.exp(lst))


def cppx_logits(scores, corpus):
  nw = len(corpus.split()) + 1
  nc = len(corpus)
  ip = -np.sum(np.log10(scores)) / nc
  print(ip, 10 ** ip)
  return 10**(ip * nc / nw)

def cppx_logprob(scores, corpus, convert=True):
  nw = len(corpus.split()) + 1
  nc = len(corpus)
  ip = -np.sum((ln2lt(scores) if convert else scores)) / nc
  print(ip, 10 **ip)
  return 10**(ip * nc / nw)

gPATH = './slgru_epoch120.model'
rPATH = './slrnn_epoch90.model'

gru = RNN1(ALPHASIZE, INTERNALSIZE, ALPHASIZE)
rnn = RNN2(ALPHASIZE, INTERNALSIZE, ALPHASIZE)

# scores = get_scores(s)

# gsc = pt_scores(s, gru, gPATH)
# rsc = pt_scores(s, rnn, rPATH)

