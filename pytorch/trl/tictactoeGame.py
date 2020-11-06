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

    def gameLoop(self)
        tictactoe = TicTacGame()
        tictactoe.getNextState()
        while (not tictactoe.isBoardFilled()):
            tictactoe.getNextState()
            if (tictactoe.isWinState()):
                break
        if (tictactoe.isWinState()):
            print(("O" if tictactoe.XTurnToPlay else "X") , "wins");
        else
        Console.WriteLine("game was a draw");
        tictactoe.printState();
        }
        public
        void
        reset()
        {
            this.gameState = new
        String[]
        {"-", "-", "-", "-", "-", "-", "-", "-", "-"};
        this.XTurnToPlay = true;
        this.winner = "TicTacToe Demo";
        this.windex = -1;
        }
        public
        void
        gamePlay()
        {
        if (this.isWinState() | | this.isBoardFilled())
        this.reset();
        this.getNextState();
        }
        public
        void
        getNextState()
        {
            int
        v = _next(0, 9);
        while (this.gameState[v] != "-") {
        v = _next(0, 9);
        }
        this.gameState[v] = this.XTurnToPlay ? "X": "O";
        this.XTurnToPlay = !this.XTurnToPlay;
        this.winner = this.isWinState()?(this.XTurnToPlay?"O":"X") + " wins": (this.isBoardFilled()
                                                                               ?"game was a draw":this.winner);
        // print('this.windex=${this.windex}');
        // this.testWinState();

    }

    public
    bool
    isWinState()
    {
    var
    winstate = false;
    for (var i=0;i < this.winstates.Length;i++){
    if (gameState[winstates[i][0]] != "-" & & gameState[winstates[i][0]] == gameState[winstates[i][1]] & & gameState[
        winstates[i][1]] == gameState[winstates[i][2]]){
    this.windex=i;
    winstate=true;
    break;
    }
    }
    return winstate;

}

public
bool
isBoardFilled()
{
return !this.gameState.ToList().Contains("-");
}

public
void
printState()
{
var
sb = new
StringBuilder();
for (var i=0;i < 3;i++){
for (var j=0;j < 3;j++)
sb.Append(this.gameState[i * 3+j]);
Console.WriteLine(sb.ToString());
sb.Clear();
}
}

/ **
*Generates
a
positive
random
integer
uniformly
distributed
on
the
range
*
from

[min], inclusive, to[max], exclusive.
* /
private
int
_next(int
min, int
max) = > min + _random.Next(max - min);

public
void
testWinState()
{
for (var i=0;i < this.winstates.Length;i++){
                                           // Console.WriteLine( @ "${this.winstates[i][0]}=${this.gameState[this.winstates[i][0]]}:${this.gameState[this.winstates[i][0]]!=" - "}");
// Console.WriteLine( @ "${this.winstates[i][1]}=${this.gameState[this.winstates[i][1]]}:${this.gameState[this.winstates[i][1]]!="-"}");
// Console.WriteLine( @ "${this.winstates[i][2]}=${this.gameState[this.winstates[i][2]]}:${this.gameState[this.winstates[i][2]]!="-"}");
}
}

def forward(self, x):
        x = self.dl1(x)
        x = torch.relu(x)

        x = self.dl2(x)
        x = torch.relu(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x