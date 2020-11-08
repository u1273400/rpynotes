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
        self.gameHistory = []
        self.qTrain = []
        self.qVal = []
        self.qTest = []
        self.lTrain = []
        self.lVal = []
        self.lTest = []
        self.gameCount = 0
        self.pindex = -1;

    def gameLoop(self):
        self.reset()
        self.getNextState()
        while not self.isBoardFilled():
            self.getNextState()
            if self.isWinState():
                break
        if self.isWinState():
            print(("O" if self.XTurnToPlay else "X"), "wins")
        else:
            print("game was a draw")
        self.printState()
        self.qUpdate()
        print("game count =", self.gameCount)

    def qUpdate(self):
        qHist = []
        if not self.isBoardFilled() and self.XTurnToPlay:
            return;
        for i in range(len(self.gameHistory)):
            qState = []
            gState = self.gameHistory.pop()
            for gp in gState:
                if gp == 'X':
                    qState.append((1, 0))
                elif gp == 'O':
                    qState.append((-1, 0))
                elif gp == '-':
                    qState.append((0, 0))
                else:
                    qState.append((1, 1))
            qHist = [[f for f, c in qState]
                , [c for f, c in qState]]
            # print(qHist[0])
            if self.gameCount % 5 == 0 and self.gameCount % 45 != 0:
                self.qTest.append(qHist[0])
                self.lTest.append(qHist[1])
            elif self.gameCount % 9 == 0:
                self.qVal.append(qHist[0])
                self.lVal.append(qHist[1])
            elif self.gameCount % 5 != 0 or self.gameCount % 9 != 0:
                self.qTrain.append(qHist[0])
                self.lTrain.append(qHist[1])
        self.gameCount += 1

    def reset(self):
        self.gameState = ["-", "-", "-", "-", "-", "-", "-", "-", "-"]
        self.XTurnToPlay = True
        self.winner = "TicTacToe Demo"
        self.windex = -1
        self.gameHistory = []
        self.pindex = -1

    def gamePlay(self):
        if self.isWinState() or self.isBoardFilled():
            self.reset()
        self.getNextState()

    def getNextState(self):
        v = random.randint(0, 8)
        while self.gameState[v] != "-":
            v = random.randint(0, 8)
        if self.XTurnToPlay:
            self.gameState[v] = '*'
            # print(self.gameState)
            self.gameHistory.append([x for x in self.gameState])
            self.gameState[v] = 'X'
        else:
            self.gameState[v] = "O"
        self.XTurnToPlay = not self.XTurnToPlay
        self.winner = (("O" if self.XTurnToPlay else "X") + " wins") if self.isWinState() else (
            "game was a draw" if self.isBoardFilled() else self.winner)
        # this.testWinState();

    def isWinState(self):
        winstate = False
        for i in range(len(self.winstates)):
            if (self.gameState[self.winstates[i][0]] != "-" and self.gameState[self.winstates[i][0]]
                    == self.gameState[self.winstates[i][1]] and
                    self.gameState[self.winstates[i][1]] == self.gameState[self.winstates[i][2]]):
                self.windex = i
                winstate = True
                break
        return winstate

    def isBoardFilled(self):
        return "-" not in self.gameState

    def printState(self):
        sb = ""
        for i in range(3):
            for j in range(3):
                sb += self.gameState[i * 3 + j]
            print(sb)
            sb = ""

    @staticmethod
    def minibatch():
        ttt=TicTacGame()
        for i in range(50):
            ttt.gameLoop()
        return ttt.qTrain, ttt.lTrain, ttt.qVal, ttt.lVal, ttt.qTest, ttt.lTest

def main():
    ttt = TicTacGame()
    ttt.gameLoop()


if __name__ == '__main__':
    main()
