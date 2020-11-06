class TicTacGame():
    def __init__(self):
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