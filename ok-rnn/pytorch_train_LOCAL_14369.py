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
import json
import torch
import datetime

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
VALI_SEQLEN = 30

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

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.GRUCell(input_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input,hidden)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(BATCHSIZE, self.hidden_size, device=device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using gpu..' if torch.cuda.is_available() else 'using cpu..')
rnn = RNN(ALPHASIZE, INTERNALSIZE, ALPHASIZE)
rnn.to(device)
  
criterion = nn.NLLLoss()
# training fn
learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()
    
    lint = []
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
        lint.append(output)
    input = torch.stack(lint).transpose(0,1).transpose(1,2)
#     print(f'is={input.size()}, cs={category_tensor.size()}')
#     print(f'is[1:]={input.size()[1:]}, cs[1:]={category_tensor.size()[1:]}')
#     print(f'is[2:]={input.size()[2:]}, cs[2:]={category_tensor.size()[2:]}')
    loss = criterion(input, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return torch.stack(lint).transpose(0,1), loss.item()
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

print_every = 250
plot_every = 100

# Keep track of losses for plotting
current_loss = 0
all_losses = []
vloss = []
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
            elapsed_time = time.time() - start
            tss = str(datetime.timedelta(seconds=elapsed_time)) # time since start string
            if epoch > 0:
                speed = epoch/elapsed_time
                eta = (nb_epoch-epoch)/speed
                sspeed = speed*60*60
                seta = str(datetime.timedelta(seconds=int(eta)))
                stats = f'average epoch rate per hr = %3.2f,  eta = {seta}'%(sspeed)
            else:
                stats ='initialising stats..'
            correct = '✓' if guess[i] == category[i] else '✗ %s' % stats 
            print('epoch %d of %d (%s) %.4f %s / %s %s' % (epoch+1, nb_epoch, tss, loss, lines[i], guess[0], correct))
        PATH = './slgru_epoch120.model'
        torch.save(rnn.state_dict(), PATH)

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
        vali_x, vali_y, _ = next(txt.rnn_minibatch_sequencer(valitext, BATCHSIZE, VALI_SEQLEN, 1))  # all data in 1 batch
        line_tensor = mb2t(vali_x)
        output, loss = train(torch.tensor(vali_y, device=device, dtype=torch.long), line_tensor)
        vloss.append(loss)
        with open('vloss.json', 'w') as f:
          json.dump(str({"vloss":vloss,"tloss":all_losses}),f)
    iter += 1
    
with open('pytorch_train.json', 'w') as f:
    json.dump(vloss, f)

