import random

class TicTacGame():
    def __init__(self):
        self.winstates = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]
        self.gameState = ["-", "-", "-", "-", "-", "-", "-", "-", "-"]
        self.XTurnToPlay = True
        self.winner = "TicTacToe Demo"
        self.windex = -1

    def gameLoop(self):
        tictactoe = TicTacGame()
        tictactoe.getNextState()
        while (not tictactoe.isBoardFilled()):
            tictactoe.getNextState()
            if (tictactoe.isWinState()):
                break
        if (tictactoe.isWinState()):
            print(("O" if tictactoe.XTurnToPlay else "X") , "wins");
        else
            print("game was a draw")
            tictactoe.printState()

    def reset(self):
        self.gameState = ["-", "-", "-", "-", "-", "-", "-", "-", "-"]
        self.XTurnToPlay = True
        self.winner = "TicTacToe Demo"
        self.windex = -1

    def gamePlay(self):
        if (self.isWinState() or self.isBoardFilled())
        self.reset()
        self.getNextState()

    def getNextState(self):
        v = random.ranint(0, 9)
        while (self.gameState[v] != "-"):
            v = random.randint(0, 9)
        self.gameState[v] =  "X" if self.XTurnToPlay else "O"
        self.XTurnToPlay = not self.XTurnToPlay
        self.winner = (("O" if self.XTurnToPlay else "X") + " wins") if self.isWinState() else ("game was a draw" if self.isBoardFilled() else self.winner)
        # print('this.windex=${this.windex}');
        # this.testWinState();

    def isWinState(self):
        winstate = False
        for ( i in  range(len(self.winstates))):
            if (self.gameState[self.winstates[i][0]] != "-" and self.gameState[self.winstates[i][0]]
                    == self.gameState[self.winstates[i][1]] and
                    self.gameState[self.winstates[i][1]] == self.gameState[self.winstates[i][2]]):
                self.windex=i
                winstate=True
                break
        return winstate

        def isBoardFilled(self):
            return !this.gameState.ToList().Contains("-");

    def printState():
        sb = new
        StringBuilder();
        for (var i=0;i < 3;i++){
        for (var j=0;j < 3;j++)
        sb.Append(this.gameState[i * 3+j]);
        Console.WriteLine(sb.ToString());
        sb.Clear();

        '''
    def testWinState(self)
{
for (var i=0;i < this.winstates.Length;i++){
                                           // Console.WriteLine( @ "${this.winstates[i][0]}=${this.gameState[this.winstates[i][0]]}:${this.gameState[this.winstates[i][0]]!=" - "}");
// Console.WriteLine( @ "${this.winstates[i][1]}=${this.gameState[this.winstates[i][1]]}:${this.gameState[this.winstates[i][1]]!="-"}");
// Console.WriteLine( @ "${this.winstates[i][2]}=${this.gameState[this.winstates[i][2]]}:${this.gameState[this.winstates[i][2]]!="-"}");
}
}

'''