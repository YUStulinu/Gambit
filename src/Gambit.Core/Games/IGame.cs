namespace Gambit.Core.Games;

/// <summary>Outcome of a position, from the point of view of the side to move.</summary>
public enum Outcome : sbyte
{
    Ongoing = 0,
    Loss = -1,   // the opponent's last move won the game
    Draw = 2,
}

/// <summary>
/// A two-player, zero-sum, perfect-information board game, implemented as a small mutable struct.
///
/// Everything the search needs is here: legal moves, playing a move, the result, and the
/// encoding of the position as input planes for the network. The static members describe the
/// game itself (board size, action space, symmetries), so the search, the self-play loop and
/// the network are written once, generically, and the JIT specialises them per game with no
/// virtual calls or allocations on the hot path.
/// </summary>
public interface IGame<TSelf> where TSelf : struct, IGame<TSelf>
{
    /// <summary>Short identifier, e.g. "connect4".</summary>
    static abstract string Name { get; }

    /// <summary>Board width and height (cells), as seen by the network.</summary>
    static abstract int Width { get; }
    static abstract int Height { get; }

    /// <summary>Size of the policy vector: one entry per possible action.</summary>
    static abstract int ActionCount { get; }

    /// <summary>Number of input planes produced by <see cref="Encode"/>.</summary>
    static abstract int PlaneCount { get; }

    /// <summary>Number of board symmetries (1 = none). Used for data augmentation.</summary>
    static abstract int SymmetryCount { get; }

    /// <summary>Upper bound on the length of a game, in moves.</summary>
    static abstract int MaxGameLength { get; }

    /// <summary>The empty starting position.</summary>
    static abstract TSelf Initial { get; }

    /// <summary>Maps an action through symmetry <paramref name="sym"/> (0 = identity).</summary>
    static abstract int TransformAction(int action, int sym);

    /// <summary>Maps a cell index (y * Width + x) through symmetry <paramref name="sym"/>.</summary>
    static abstract int TransformCell(int cell, int sym);

    /// <summary>Moves played so far.</summary>
    int Ply { get; }

    /// <summary>0 if the first player is to move, 1 otherwise.</summary>
    int SideToMove { get; }

    /// <summary>The result for the side to move (Loss means the previous move won).</summary>
    Outcome Outcome { get; }

    bool IsLegal(int action);

    /// <summary>Writes the legal actions into <paramref name="buffer"/> and returns how many there are.</summary>
    int LegalMoves(Span<int> buffer);

    /// <summary>Plays a legal action for the side to move.</summary>
    void Play(int action);

    /// <summary>
    /// Writes PlaneCount x Height x Width floats describing the position from the point of view
    /// of the side to move (own stones first), the layout the network expects.
    /// </summary>
    void Encode(Span<float> planes);

    /// <summary>A hash of the position (equal positions have equal hashes).</summary>
    ulong Hash { get; }

    /// <summary>The stone at (x, y), y = 0 is the top row: 0 empty, 1 first player, 2 second player.</summary>
    int CellAt(int x, int y);

    /// <summary>Human-readable name of an action (e.g. "d" for a Connect Four column, "e5" in Gomoku).</summary>
    string ActionName(int action);

    /// <summary>Parses an action name, or returns -1.</summary>
    int ParseAction(string text);
}

public static class GameExtensions
{
    /// <summary>Value of a finished position for the side to move: -1, 0 or +1.</summary>
    public static float TerminalValue(this Outcome o) => o switch
    {
        Outcome.Loss => -1f,
        Outcome.Draw => 0f,
        _ => throw new InvalidOperationException("the game is not over"),
    };

    public static bool IsOver(this Outcome o) => o != Outcome.Ongoing;

    /// <summary>A text drawing of the board.</summary>
    public static string Render<TGame>(this TGame g) where TGame : struct, IGame<TGame>
    {
        var sb = new System.Text.StringBuilder();
        for (int y = 0; y < TGame.Height; y++)
        {
            for (int x = 0; x < TGame.Width; x++)
                sb.Append(g.CellAt(x, y) switch { 1 => " X", 2 => " O", _ => " ." });
            sb.AppendLine();
        }
        return sb.ToString();
    }
}
