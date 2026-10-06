using System.Runtime.CompilerServices;

namespace Gambit.Core.Games;

/// <summary>
/// Gomoku on a 9x9 board: five (or more) in a row wins, in any of the four directions.
///
/// Each player's stones are one 128-bit bitboard with a row stride of 10: cell (x, y) is bit
/// 10y + x, and bit 10y + 9 is an always-empty sentinel column, so shifting by 1 (horizontal),
/// 10 (vertical), 11 (diagonal \) or 9 (diagonal /) never wraps a line from one row into the
/// next. Five in a row is then three shift-and-AND steps per direction over the whole board.
/// Actions are cell indices y * 9 + x.
/// </summary>
public struct Gomoku : IGame<Gomoku>
{
    public const int Size = 9;
    const int Stride = Size + 1;
    const int Cells = Size * Size;

    UInt128 _black;   // first player
    UInt128 _white;
    byte _moves;
    Outcome _outcome;

    public static string Name => "gomoku";
    public static int Width => Size;
    public static int Height => Size;
    public static int ActionCount => Cells;
    public static int PlaneCount => 3;
    public static int SymmetryCount => 8;
    public static int MaxGameLength => Cells;
    public static Gomoku Initial => default;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static UInt128 Bit(int action) => UInt128.One << (action / Size * Stride + action % Size);

    /// <summary>Dihedral symmetries of the square: 4 rotations, each optionally mirrored.</summary>
    public static int TransformCell(int cell, int sym)
    {
        int x = cell % Size, y = cell / Size, n = Size - 1;
        if ((sym & 4) != 0) x = n - x;
        for (int i = 0; i < (sym & 3); i++) (x, y) = (n - y, x);
        return y * Size + x;
    }

    public static int TransformAction(int action, int sym) => TransformCell(action, sym);

    public readonly int Ply => _moves;
    public readonly int SideToMove => _moves & 1;
    public readonly Outcome Outcome => _outcome;

    public readonly ulong Hash
    {
        get
        {
            ulong a = (ulong)_black, b = (ulong)(_black >> 64), c = (ulong)_white, d = (ulong)(_white >> 64);
            ulong h = a * 0x9E3779B97F4A7C15UL ^ b * 0xC2B2AE3D27D4EB4FUL ^ c * 0x165667B19E3779F9UL ^ d * 0xD6E8FEB86659FD93UL;
            h ^= h >> 31;
            h *= 0x94D049BB133111EBUL;
            return h ^ (h >> 29);
        }
    }

    public readonly bool IsLegal(int action) =>
        (uint)action < Cells && _outcome == Outcome.Ongoing && ((_black | _white) & Bit(action)) == UInt128.Zero;

    /// <summary>Every cell of the board in the stride-10 layout (the sentinel column excluded).</summary>
    static readonly UInt128 BoardMask = MakeBoardMask();

    static UInt128 MakeBoardMask()
    {
        UInt128 m = UInt128.Zero;
        for (int a = 0; a < Cells; a++) m |= Bit(a);
        return m;
    }

    public readonly int LegalMoves(Span<int> buffer)
    {
        if (_outcome != Outcome.Ongoing) return 0;
        // Walk the set bits of the empty cells instead of testing all 81.
        UInt128 empty = BoardMask & ~(_black | _white);
        int n = 0;
        ulong lo = (ulong)empty, hi = (ulong)(empty >> 64);
        while (lo != 0)
        {
            int bit = System.Numerics.BitOperations.TrailingZeroCount(lo);
            lo &= lo - 1;
            buffer[n++] = bit / Stride * Size + bit % Stride;
        }
        while (hi != 0)
        {
            int bit = 64 + System.Numerics.BitOperations.TrailingZeroCount(hi);
            hi &= hi - 1;
            buffer[n++] = bit / Stride * Size + bit % Stride;
        }
        return n;
    }

    public void Play(int action)
    {
        bool black = (_moves & 1) == 0;
        if (black) _black |= Bit(action); else _white |= Bit(action);
        _moves++;
        if (HasFive(black ? _black : _white)) _outcome = Outcome.Loss;
        else if (_moves == Cells) _outcome = Outcome.Draw;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static bool Five(UInt128 b, int d)
    {
        UInt128 m = b & (b >> d);          // runs of 2
        m &= m >> (2 * d);                 // runs of 4
        return (m & (b >> (4 * d))) != UInt128.Zero;
    }

    public static bool HasFive(UInt128 b) => Five(b, 1) || Five(b, Stride) || Five(b, Stride + 1) || Five(b, Stride - 1);

    public readonly void Encode(Span<float> planes)
    {
        planes[..(PlaneCount * Cells)].Clear();
        bool blackToMove = (_moves & 1) == 0;
        UInt128 own = blackToMove ? _black : _white, opp = blackToMove ? _white : _black;
        for (int a = 0; a < Cells; a++)
        {
            UInt128 bit = Bit(a);
            if ((own & bit) != UInt128.Zero) planes[a] = 1f;
            else if ((opp & bit) != UInt128.Zero) planes[Cells + a] = 1f;
        }
        if (!blackToMove) planes.Slice(2 * Cells, Cells).Fill(1f);
    }

    public readonly int CellAt(int x, int y)
    {
        UInt128 bit = Bit(y * Size + x);
        if ((_black & bit) != UInt128.Zero) return 1;
        return (_white & bit) != UInt128.Zero ? 2 : 0;
    }

    public readonly string ActionName(int action) => $"{(char)('a' + action % Size)}{Size - action / Size}";

    public readonly int ParseAction(string text)
    {
        if (text.Length < 2) return -1;
        int x = char.ToLowerInvariant(text[0]) - 'a';
        if (!int.TryParse(text.AsSpan(1), out int row)) return -1;
        int y = Size - row;
        return x is >= 0 and < Size && y is >= 0 and < Size ? y * Size + x : -1;
    }

    public override readonly string ToString() => this.Render();
}
