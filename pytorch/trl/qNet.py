import torch
from torch import nn
import torch.optim as optim
from pytorch.trl.qGame import TicTacGame

INPUT_SIZE = 9
OUTPUT_SIZE = 9


class TicTacNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dl1 = nn.Linear(INPUT_SIZE, 36)
        self.dl2 = nn.Linear(36, 36)
        self.output_layer = nn.Linear(36, OUTPUT_SIZE)

    def forward(self, x):
        x = self.dl1(x)
        x = torch.relu(x)

        x = self.dl2(x)
        x = torch.relu(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def train():
        net=TicTacNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        testdata = []
        testlabels = []
        running_loss = 0.0
        for i in range(2):  # loop over the dataset multiple times

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, qVal, lVal, qTest, lTest = TicTacGame.minibatch()
            testdata += qTest
            testlabels += lTest

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2 == 1:    # print every 2 mini-batches
                print('[mini-batch %5d] loss: %.3f' %
                      (i + 1, running_loss / (i+1)))

        print('Finished Training')
