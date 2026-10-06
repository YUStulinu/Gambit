using Gambit.Core.Games;
using Gambit.Core.Neural;
using Gambit.Core.Search;

namespace Gambit.Core.Session;

/// <summary>A snapshot of the AI's search, for display.</summary>
public sealed record Analysis(float RootValue, int Simulations, double Seconds, IReadOnlyList<MoveStats> Moves, IReadOnlyList<int> PrincipalVariation)
{
    /// <summary>Probability-like score for the side to move: (value + 1) / 2.</summary>
    public double WinChance => (RootValue + 1) / 2;
    public int TotalVisits => Moves.Sum(m => m.Visits);
}

/// <summary>
/// One game in progress, for the desktop app: the position, the move history (with undo), and an
/// AI player built on <see cref="MctsPlayer{TGame}"/>. Non-generic, so the UI does not need to
/// know which game it is showing.
/// </summary>
public interface IGameSession
{
    string GameName { get; }
    string Title { get; }
    int Width { get; }
    int Height { get; }
    /// <summary>Connect Four: actions are columns (a click anywhere in a column plays it).</summary>
    bool ColumnGame { get; }
    int WinLength { get; }
    int Ply { get; }
    int SideToMove { get; }
    bool IsOver { get; }
    /// <summary>1 if the first player won, 2 the second, 0 a draw, -1 still playing.</summary>
    int Winner { get; }
    int CellAt(int x, int y);
    IReadOnlyList<int> Moves { get; }
    /// <summary>The cell (x, y) where the stone of the last move is, or null.</summary>
    (int X, int Y)? LastCell { get; }
    /// <summary>The cells of the winning line, if the game was won.</summary>
    IReadOnlyList<(int X, int Y)> WinningLine { get; }
    string NetworkDescription { get; }

    /// <summary>The action a click on (x, y) means, or -1 if it is not a legal move.</summary>
    int ActionAt(int x, int y);
    /// <summary>The cell where an action's stone goes (for overlays): the landing cell for Connect Four.</summary>
    (int X, int Y) CellOf(int action);
    string ActionName(int action);
    bool IsLegal(int action);
    void Play(int action);
    void Undo();
    void Reset();

    /// <summary>Searches the current position; <paramref name="progress"/> receives snapshots while it runs.</summary>
    Task<Analysis> ThinkAsync(int simulations, Action<Analysis>? progress, CancellationToken cancel);
}

public sealed class GameSession<TGame> : IGameSession where TGame : struct, IGame<TGame>
{
    readonly List<int> _moves = [];
    readonly List<TGame> _history = [];
    readonly MctsPlayer<TGame> _ai;
    TGame _state = TGame.Initial;

    public GameSession(IEvaluator<TGame> evaluator, string networkDescription, string title, int winLength)
    {
        _ai = new MctsPlayer<TGame>(evaluator, 400, batch: 16, seed: Environment.TickCount);
        NetworkDescription = networkDescription;
        Title = title;
        WinLength = winLength;
    }

    public string GameName => TGame.Name;
    public string Title { get; }
    public int Width => TGame.Width;
    public int Height => TGame.Height;
    public bool ColumnGame => typeof(TGame) == typeof(ConnectFour);
    public int WinLength { get; }
    public int Ply => _state.Ply;
    public int SideToMove => _state.SideToMove;
    public bool IsOver => _state.Outcome.IsOver();
    public string NetworkDescription { get; }
    public IReadOnlyList<int> Moves => _moves;

    public int Winner => _state.Outcome switch
    {
        Outcome.Ongoing => -1,
        Outcome.Draw => 0,
        _ => (_state.Ply - 1) % 2 == 0 ? 1 : 2,
    };

    public int CellAt(int x, int y) => _state.CellAt(x, y);

    public (int X, int Y)? LastCell
    {
        get
        {
            if (_moves.Count == 0) return null;
            var before = _history[^1];
            for (int y = 0; y < Height; y++)
                for (int x = 0; x < Width; x++)
                    if (before.CellAt(x, y) == 0 && _state.CellAt(x, y) != 0) return (x, y);
            return null;
        }
    }

    public IReadOnlyList<(int X, int Y)> WinningLine
    {
        get
        {
            if (_state.Outcome != Outcome.Loss || LastCell is not { } last) return [];
            int stone = _state.CellAt(last.X, last.Y);
            foreach (var (dx, dy) in new[] { (1, 0), (0, 1), (1, 1), (1, -1) })
            {
                var line = new List<(int, int)> { last };
                for (int s = 1; Inside(last.X + s * dx, last.Y + s * dy) && _state.CellAt(last.X + s * dx, last.Y + s * dy) == stone; s++)
                    line.Add((last.X + s * dx, last.Y + s * dy));
                for (int s = 1; Inside(last.X - s * dx, last.Y - s * dy) && _state.CellAt(last.X - s * dx, last.Y - s * dy) == stone; s++)
                    line.Add((last.X - s * dx, last.Y - s * dy));
                if (line.Count >= WinLength) return line;
            }
            return [];
        }
    }

    bool Inside(int x, int y) => x >= 0 && y >= 0 && x < Width && y < Height;

    public int ActionAt(int x, int y)
    {
        if (!Inside(x, y)) return -1;
        int a = ColumnGame ? x : y * Width + x;
        return _state.IsLegal(a) ? a : -1;
    }

    public (int X, int Y) CellOf(int action)
    {
        if (!ColumnGame) return (action % Width, action / Width);
        int y = Height - 1;
        while (y >= 0 && _state.CellAt(action, y) != 0) y--;
        return (action, Math.Max(y, 0));
    }

    public string ActionName(int action) => _state.ActionName(action);
    public bool IsLegal(int action) => _state.IsLegal(action);

    public void Play(int action)
    {
        if (!_state.IsLegal(action)) throw new InvalidOperationException($"illegal move {action}");
        _history.Add(_state);
        _moves.Add(action);
        _state.Play(action);
    }

    public void Undo()
    {
        if (_moves.Count == 0) return;
        _state = _history[^1];
        _history.RemoveAt(_history.Count - 1);
        _moves.RemoveAt(_moves.Count - 1);
    }

    public void Reset()
    {
        _state = TGame.Initial;
        _moves.Clear();
        _history.Clear();
        _ai.NewGame();
    }

    public Task<Analysis> ThinkAsync(int simulations, Action<Analysis>? progress, CancellationToken cancel)
    {
        var state = _state;
        return Task.Run(() =>
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            long lastReport = -1000;
            Analysis Snapshot(SearchTree<TGame> t) =>
                new(t.RootValue, t.RootVisits, sw.Elapsed.TotalSeconds, t.RootStats(), t.PrincipalVariation());
            var r = _ai.Search(state, simulations, cancel, tree =>
            {
                if (progress == null || sw.ElapsedMilliseconds - lastReport < 50) return;
                lastReport = sw.ElapsedMilliseconds;
                progress(Snapshot(tree));
            });
            return Snapshot(_ai.Tree) with { Seconds = r.Seconds };
        }, CancellationToken.None);
    }
}

/// <summary>Creates sessions, with the trained network when one is available.</summary>
public static class Sessions
{
    public static IGameSession Create(string game, string? modelPath)
    {
        NativeNetwork? net = modelPath != null && File.Exists(modelPath) ? NativeNetwork.Load(modelPath) : null;
        string desc = net != null ? $"{net.Spec} network ({Path.GetFileName(modelPath)})" : "no trained network found: pure MCTS with random rollouts";
        return game switch
        {
            "connect4" => new GameSession<ConnectFour>(
                net?.For<ConnectFour>() ?? new RolloutEvaluator<ConnectFour>(4), desc, "Connect Four", 4),
            "gomoku" => new GameSession<Gomoku>(
                net?.For<Gomoku>() ?? new RolloutEvaluator<Gomoku>(2), desc, "Gomoku 9x9", 5),
            _ => throw new ArgumentException($"unknown game {game}"),
        };
    }
}
