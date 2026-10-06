using System.Numerics;
using System.Runtime.CompilerServices;

namespace Gambit.Core.Games;

/// <summary>
/// Connect Four (7 columns x 6 rows) on two 64-bit bitboards.
///
/// Bit layout (the one used by Pascal Pons' solver): column c occupies bits 7c .. 7c+6, row 0
/// (bottom) first, with bit 7c+6 a permanently empty sentinel that stops alignments from
/// wrapping between columns.
///
///     .  .  .  .  .  .  .     6 13 20 27 34 41 48   <- sentinel row
///     5 12 19 26 33 40 47     row 5 (top)
///     ...
///     0  7 14 21 28 35 42     row 0 (bottom)
///
/// <c>mask</c> holds every stone and <c>current</c> the stones of the side to move. Playing in a
/// column is one addition (<c>mask + bottom</c> carries up to the first empty cell), and a win is
/// detected with four shift-and-AND pairs over the whole board at once.
/// </summary>
public struct ConnectFour : IGame<ConnectFour>
{
    public const int Columns = 7;
    public const int Rows = 6;
    const int Stride = Rows + 1;

    ulong _current;   // stones of the side to move
    ulong _mask;      // all stones
    byte _moves;
    Outcome _outcome;

    public static string Name => "connect4";
    public static int Width => Columns;
    public static int Height => Rows;
    public static int ActionCount => Columns;
    public static int PlaneCount => 3;
    public static int SymmetryCount => 2;
    public static int MaxGameLength => Columns * Rows;
    public static ConnectFour Initial => default;

    public static int TransformAction(int action, int sym) => sym == 0 ? action : Columns - 1 - action;

    public static int TransformCell(int cell, int sym)
    {
        if (sym == 0) return cell;
        int y = cell / Columns, x = cell % Columns;
        return y * Columns + (Columns - 1 - x);
    }

    public readonly int Ply => _moves;
    public readonly int SideToMove => _moves & 1;
    public readonly Outcome Outcome => _outcome;
    public readonly ulong Mask => _mask;
    public readonly ulong Current => _current;

    /// <summary>Unique 49-bit key of the position (current + mask identifies it exactly).</summary>
    public readonly ulong Key => _current + _mask;

    public readonly ulong Hash
    {
        get
        {
            ulong h = Key * 0x9E3779B97F4A7C15UL;
            return h ^ (h >> 29);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong TopMask(int col) => 1UL << (Rows - 1 + col * Stride);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong BottomMask(int col) => 1UL << (col * Stride);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static ulong ColumnMask(int col) => ((1UL << Rows) - 1) << (col * Stride);

    /// <summary>Bottom cell of every column.</summary>
    public const ulong BottomRow = 0b0000001_0000001_0000001_0000001_0000001_0000001_0000001UL;

    /// <summary>Every playable cell (excludes the sentinel row).</summary>
    public const ulong BoardMask = BottomRow * ((1UL << Rows) - 1);

    public readonly bool IsLegal(int action) =>
        (uint)action < Columns && _outcome == Outcome.Ongoing && (_mask & TopMask(action)) == 0;

    public readonly int LegalMoves(Span<int> buffer)
    {
        if (_outcome != Outcome.Ongoing) return 0;
        int n = 0;
        for (int c = 0; c < Columns; c++)
            if ((_mask & TopMask(c)) == 0) buffer[n++] = c;
        return n;
    }

    public void Play(int action)
    {
        _current ^= _mask;
        _mask |= _mask + BottomMask(action);
        _moves++;
        // The stones of the player who just moved are now mask ^ current.
        if (HasFour(_current ^ _mask)) _outcome = Outcome.Loss;
        else if (_moves == Columns * Rows) _outcome = Outcome.Draw;
    }

    /// <summary>True if the bitboard contains four aligned stones.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool HasFour(ulong p)
    {
        ulong m = p & (p >> 1);                     // vertical
        if ((m & (m >> 2)) != 0) return true;
        m = p & (p >> Stride);                      // horizontal
        if ((m & (m >> (2 * Stride))) != 0) return true;
        m = p & (p >> (Stride - 1));                // diagonal /
        if ((m & (m >> (2 * (Stride - 1)))) != 0) return true;
        m = p & (p >> (Stride + 1));                // diagonal \
        return (m & (m >> (2 * (Stride + 1)))) != 0;
    }

    /// <summary>True if playing <paramref name="col"/> wins immediately for the side to move.</summary>
    public readonly bool IsWinningMove(int col) =>
        HasFour(_current | ((_mask + BottomMask(col)) & ColumnMask(col)));

    /// <summary>True if the opponent would win by playing <paramref name="col"/> (a threat the side to move must block).</summary>
    public readonly bool IsOpponentWinningMove(int col) =>
        (_mask & TopMask(col)) == 0 && HasFour((_current ^ _mask) | ((_mask + BottomMask(col)) & ColumnMask(col)));

    /// <summary>Plays a sequence of columns given as digits 1-7, e.g. "4453".</summary>
    public static ConnectFour FromMoves(string moves)
    {
        var g = Initial;
        foreach (char ch in moves)
        {
            int c = ch - '1';
            if (!g.IsLegal(c)) throw new ArgumentException($"illegal move '{ch}' in \"{moves}\"");
            g.Play(c);
        }
        return g;
    }

    public readonly void Encode(Span<float> planes)
    {
        const int area = Columns * Rows;
        planes[..(PlaneCount * area)].Clear();
        ulong own = _current, opp = _current ^ _mask;
        for (int c = 0; c < Columns; c++)
        {
            for (int r = 0; r < Rows; r++)
            {
                ulong bit = 1UL << (c * Stride + r);
                int cell = (Rows - 1 - r) * Columns + c;   // y = 0 is the top row
                if ((own & bit) != 0) planes[cell] = 1f;
                else if ((opp & bit) != 0) planes[area + cell] = 1f;
            }
        }
        // Whose turn it is matters in Connect Four (parity decides many endgames).
        if ((_moves & 1) == 1) planes.Slice(2 * area, area).Fill(1f);
    }

    public readonly int CellAt(int x, int y)
    {
        int r = Rows - 1 - y;
        ulong bit = 1UL << (x * Stride + r);
        if ((_mask & bit) == 0) return 0;
        ulong first = (_moves & 1) == 0 ? _current : _current ^ _mask;
        return (first & bit) != 0 ? 1 : 2;
    }

    /// <summary>Height of the stack in a column (0..6).</summary>
    public readonly int ColumnHeight(int col) => BitOperations.PopCount(_mask & ColumnMask(col));

    public readonly string ActionName(int action) => ((char)('1' + action)).ToString();

    public readonly int ParseAction(string text) =>
        text.Length == 1 && text[0] >= '1' && text[0] <= '7' ? text[0] - '1' : -1;

    public override readonly string ToString() => this.Render();
}
