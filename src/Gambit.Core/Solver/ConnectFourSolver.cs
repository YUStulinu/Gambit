using System.Numerics;
using Gambit.Core.Games;

namespace Gambit.Core.Solver;

/// <summary>
/// A perfect Connect Four solver (after Pascal Pons, "Solving Connect 4: how to build a perfect
/// AI"): negamax with alpha-beta pruning on bitboards, a transposition table, move ordering by the
/// number of winning threats a move creates, pruning of moves that hand the opponent an immediate
/// win, and a null-window search that narrows the score interval by bisection.
///
/// Score of a position for the side to move: 0 for a draw; positive if it wins, equal to
/// (43 - moves at the end of the game) / 2, so a faster win scores higher; negative if it loses,
/// the opposite of the opponent's score. Used to grade the AI's moves exactly.
/// </summary>
public sealed class ConnectFourSolver
{
    const int W = ConnectFour.Columns, H = ConnectFour.Rows, Cells = W * H;
    const int MinScore = -Cells / 2 + 3, MaxScore = (Cells + 1) / 2 - 3;
    const ulong Bottom = ConnectFour.BottomRow, Board = ConnectFour.BoardMask;

    static readonly int[] ColumnOrder = [3, 2, 4, 1, 5, 0, 6];   // centre first

    readonly ulong[] _keys;
    readonly sbyte[] _values;
    readonly ulong _size;

    /// <summary>Positions visited by the last <see cref="Solve"/> call.</summary>
    public long NodeCount { get; private set; }

    public ConnectFourSolver(int tableSizeLog2 = 23)
    {
        _size = (1UL << tableSizeLog2) + 9;   // odd size: the 49-bit keys spread better
        _keys = new ulong[_size];
        _values = new sbyte[_size];
    }

    public void ClearTable()
    {
        Array.Clear(_keys);
        Array.Clear(_values);
    }

    struct Pos
    {
        public ulong Current, Mask;
        public int Moves;

        public readonly ulong Key => Current + Mask;
        public readonly ulong Possible => (Mask + Bottom) & Board;
        public readonly ulong WinningCells => WinningPositions(Current, Mask);
        public readonly ulong OpponentWinningCells => WinningPositions(Current ^ Mask, Mask);
        public readonly bool CanWinNext => (WinningCells & Possible) != 0;

        public void Play(ulong move)
        {
            Current ^= Mask;
            Mask |= move;
            Moves++;
        }

        /// <summary>Moves that do not let the opponent win right away (0 if every move loses).</summary>
        public readonly ulong NonLosingMoves()
        {
            ulong possible = Possible;
            ulong oppWin = OpponentWinningCells;
            ulong forced = possible & oppWin;
            if (forced != 0)
            {
                if ((forced & (forced - 1)) != 0) return 0;   // two threats: cannot block both
                possible = forced;
            }
            return possible & ~(oppWin >> 1);                // never play right below an opponent's threat
        }

        public readonly int MoveScore(ulong move) => BitOperations.PopCount(WinningPositions(Current | move, Mask));
    }

    /// <summary>Empty cells that would complete four for the stones in <paramref name="p"/>.</summary>
    static ulong WinningPositions(ulong p, ulong mask)
    {
        // vertical
        ulong r = (p << 1) & (p << 2) & (p << 3);
        // horizontal
        ulong q = (p << (H + 1)) & (p << 2 * (H + 1));
        r |= q & (p << 3 * (H + 1));
        r |= q & (p >> (H + 1));
        q = (p >> (H + 1)) & (p >> 2 * (H + 1));
        r |= q & (p << (H + 1));
        r |= q & (p >> 3 * (H + 1));
        // diagonal 1
        q = (p << H) & (p << 2 * H);
        r |= q & (p << 3 * H);
        r |= q & (p >> H);
        q = (p >> H) & (p >> 2 * H);
        r |= q & (p << H);
        r |= q & (p >> 3 * H);
        // diagonal 2
        q = (p << (H + 2)) & (p << 2 * (H + 2));
        r |= q & (p << 3 * (H + 2));
        r |= q & (p >> (H + 2));
        q = (p >> (H + 2)) & (p >> 2 * (H + 2));
        r |= q & (p << (H + 2));
        r |= q & (p >> 3 * (H + 2));
        return r & (Board ^ mask);
    }

    static ulong ColumnMask(int c) => ((1UL << H) - 1) << (c * (H + 1));

    int Negamax(in Pos p, int alpha, int beta)
    {
        NodeCount++;
        ulong next = p.NonLosingMoves();
        if (next == 0) return -(Cells - p.Moves) / 2;
        if (p.Moves >= Cells - 2) return 0;

        int min = -(Cells - 2 - p.Moves) / 2;   // the opponent cannot win on its next move
        if (alpha < min)
        {
            alpha = min;
            if (alpha >= beta) return alpha;
        }
        int max = (Cells - 1 - p.Moves) / 2;    // we cannot win on our next move
        ulong key = p.Key;
        ulong slot = key % _size;
        if (_keys[slot] == key) max = _values[slot] + MinScore - 1;
        if (beta > max)
        {
            beta = max;
            if (alpha >= beta) return beta;
        }

        // Order moves by the number of threats they create (insertion sort, centre-first ties).
        Span<ulong> moves = stackalloc ulong[W];
        Span<int> scores = stackalloc int[W];
        int n = 0;
        for (int i = W - 1; i >= 0; i--)
        {
            ulong move = next & ColumnMask(ColumnOrder[i]);
            if (move == 0) continue;
            int s = p.MoveScore(move);
            int j = n++;
            for (; j > 0 && scores[j - 1] > s; j--)
            {
                moves[j] = moves[j - 1];
                scores[j] = scores[j - 1];
            }
            moves[j] = move;
            scores[j] = s;
        }
        for (int i = n - 1; i >= 0; i--)
        {
            var child = p;
            child.Play(moves[i]);
            int score = -Negamax(child, -beta, -alpha);
            if (score >= beta) return score;
            if (score > alpha) alpha = score;
        }
        _keys[slot] = key;
        _values[slot] = (sbyte)(alpha - MinScore + 1);   // an upper bound
        return alpha;
    }

    /// <summary>The exact score of <paramref name="game"/> for the side to move (see the class remarks).</summary>
    public int Solve(in ConnectFour game)
    {
        // Already lost: the opponent won with the move that made Ply stones, scoring (43 - (Ply - 1)) / 2.
        if (game.Outcome == Outcome.Loss) return -(Cells + 2 - game.Ply) / 2;
        if (game.Outcome == Outcome.Draw) return 0;
        NodeCount = 0;
        var p = new Pos { Current = game.Current, Mask = game.Mask, Moves = game.Ply };
        if (p.CanWinNext) return (Cells + 1 - p.Moves) / 2;
        int min = -(Cells - p.Moves) / 2;
        int max = (Cells + 1 - p.Moves) / 2;
        while (min < max)
        {
            int med = min + (max - min) / 2;
            if (med <= 0 && min / 2 < med) med = min / 2;
            else if (med >= 0 && max / 2 > med) med = max / 2;
            int r = Negamax(p, med, med + 1);
            if (r <= med) max = r;
            else min = r;
        }
        return min;
    }

    /// <summary>
    /// The score of every legal move (from the mover's point of view), or null for illegal columns.
    /// The optimal moves are those with the highest score.
    /// </summary>
    public int?[] ScoreMoves(in ConnectFour game)
    {
        var result = new int?[W];
        for (int c = 0; c < W; c++)
        {
            if (!game.IsLegal(c)) continue;
            if (game.IsWinningMove(c))
            {
                result[c] = (Cells + 1 - game.Ply) / 2;
                continue;
            }
            var child = game;
            child.Play(c);
            result[c] = child.Outcome == Outcome.Draw ? 0 : -Solve(child);
        }
        return result;
    }

    /// <summary>Win (1), draw (0) or loss (-1) for the side to move under perfect play.</summary>
    public static int Sign(int score) => Math.Sign(score);
}
