# encoding: UTF-8
# Copyright 2017 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import tensorflow as tf
# from tensorflow.contrib import layers
# from tensorflow.contrib import rnn  # rnn stuff temporarily in contrib, moving back to code in TF 1.1
import os
import time
import math
import numpy as np
import my_txtutils as txt
# tf.set_random_seed(0)

# model parameters
#
# Usage:
#   Training only:
#         Leave all the parameters as they are
#         Disable validation to run a bit faster (set validation=False below)
#         You can follow progress in Tensorboard: tensorboard --log-dir=log
#   Training and experimentation (default):
#         Keep validation enabled
#         You can now play with the parameters anf follow the effects in Tensorboard
#         A good choice of parameters ensures that the testing and validation curves stay close
#         To see the curves drift apart ("overfitting") try to use an insufficient amount of
#         training data (shakedir = "shakespeare/t*.txt" for example)
#
SEQLEN = 30
BATCHSIZE = 200
ALPHASIZE = txt.ALPHASIZE
INTERNALSIZE = 512
NLAYERS = 3
learning_rate = 0.001  # fixed learning rate
dropout_pkeep = 0.8    # some dropout
nb_epoch = 75

# load data, either shakespeare, or the Python source of Tensorflow itself
shakedir = "txts/*.txt"
#shakedir = "../tensorflow/**/*.py"
codetext, valitext, bookranges = txt.read_data_files(shakedir, validation=True)

# display some stats on the data
epoch_size = len(codetext) // (BATCHSIZE * SEQLEN)
txt.print_data_stats(len(codetext), len(valitext), epoch_size)


if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")

# model
import torch.nn as nn
import torch

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, layers):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, layers)

    def forward(self, input, hidden):
        # combined = torch.cat((input, hidden), 1)
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(NLAYERS, BATCHSIZE, self.hidden_size, device=device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gru = GRU(ALPHASIZE, INTERNALSIZE, NLAYERS)

criterion = nn.NLLLoss()
# training fn
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = gru.initHidden()
    softmax = nn.LogSoftmax(dim=1)
    fc_layer = nn.Linear(INTERNALSIZE, ALPHASIZE)        
    gru.zero_grad()
    
    output, hidden = gru(line_tensor, hidden)
    #print(f'gos={output.size()}, is={line_tensor.size()}')
    output = fc_layer(output)
    #print(f'fc={output.size()}, ls={line_tensor.size()}')
    output = softmax(output)
    #print(f'sm={output.size()}, cs[2:]={category_tensor.size()}')
    input=output.transpose(0,1).transpose(1,2)
    loss = criterion(input, category_tensor) # N (batch),C
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in gru.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output.transpose(0,1), loss.item()

# init train

def lin2txt(lt):
    return ''.join([chr(txt.convert_to_alphabet(c))  if c != 0 else '' for c in lt])

def mb2t(rows):
    rows=rows.transpose()
    tensor = torch.zeros(rows.shape[0], rows.shape[1], ALPHASIZE, device=device)
    for i, row in enumerate(rows):
        for j, letter_code in enumerate(row):
            tensor[i][j][letter_code] = 1
    return tensor

import time
import math

n_iters = 100000
print_every = 250
plot_every = 100



# Keep track of losses for plotting
current_loss = 0
all_losses = []
iter=0
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=nb_epoch):
    #category, line, category_tensor, line_tensor = randomTrainingExample()
    category =  [lin2txt(l) for l in y_]
    lines = [lin2txt(l) for l in x]
    category_tensor=mb2t(y_)
    line_tensor=mb2t(x)
    output, loss = train(torch.tensor(y_, device=device, dtype=torch.long), line_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess = [lin2txt([ch.argmax(dim=0) for ch in line]) for line in output]
        for i in range(2):
            correct = '✓' if guess[i] == category[i] else '✗ %s' % category[i] 
            print('epoch %d of  %d (%s) %.4f %s / %s %s' % (epoch, nb_epoch, timeSince(start), loss, lines[i], guess[0], correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
        vali_x, vali_y, _ = next(txt.rnn_minibatch_sequencer(valitext, BATCHSIZE, VALI_SEQLEN, 1))  # all data in 1 batch
        line_tensor = mb2t(vali_x)
        output, loss = train(torch.tensor(vali_y, device=device, dtype=torch.long), line_tensor)
        vloss.append(loss)
        plt.plot(vloss)  
    iter += 1
    
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# product = reduce((lambda x, y: x * y), [1, 2, 3, 4])

plt.figure()
plt.plot(all_losses)

