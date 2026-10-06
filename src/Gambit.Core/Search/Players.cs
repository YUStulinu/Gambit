using Gambit.Core.Games;

namespace Gambit.Core.Search;

/// <summary>A player for matches: given a position, picks a move.</summary>
public interface IPlayer<TGame> where TGame : struct, IGame<TGame>
{
    string Name { get; }
    int ChooseMove(in TGame state);
    /// <summary>Called at the start of every game.</summary>
    void NewGame() { }
}

public sealed class RandomPlayer<TGame>(int seed = 0) : IPlayer<TGame> where TGame : struct, IGame<TGame>
{
    readonly Random _rng = new(seed);
    public string Name => "random";

    public int ChooseMove(in TGame state)
    {
        Span<int> moves = stackalloc int[TGame.ActionCount];
        int n = state.LegalMoves(moves);
        return moves[_rng.Next(n)];
    }
}

/// <summary>The outcome of a search, for display or for training.</summary>
public sealed record SearchResult(int BestMove, float RootValue, int Simulations, MoveStats[] Moves, List<int> PrincipalVariation,
                                  double Seconds);

/// <summary>
/// MCTS guided by an evaluator, for one game at a time (matches, the desktop app).
///
/// Each step selects up to <c>batch</c> leaves with virtual loss, evaluates them in one call and
/// expands them, so a GPU network still gets reasonably sized batches. The tree is kept between
/// moves: when the new position is the old root after one or two moves, the matching subtree is
/// reused.
/// </summary>
public sealed class MctsPlayer<TGame> : IPlayer<TGame> where TGame : struct, IGame<TGame>
{
    readonly IEvaluator<TGame> _eval;
    readonly SearchTree<TGame> _tree;
    readonly TGame[] _states;
    readonly int[] _leaves;
    readonly float[] _logits;
    readonly float[] _values;

    public string Name { get; }
    public int Simulations { get; set; }
    public int Batch { get; }
    public float Temperature { get; set; }
    public bool ReuseTree { get; set; } = true;
    public SearchResult? LastResult { get; private set; }
    public SearchTree<TGame> Tree => _tree;

    public MctsPlayer(IEvaluator<TGame> evaluator, int simulations, int batch = 8, SearchConfig? config = null, int seed = 0,
                      string? name = null, float temperature = 0f)
    {
        _eval = evaluator;
        Simulations = simulations;
        Batch = Math.Max(1, batch);
        Temperature = temperature;
        _tree = new SearchTree<TGame>(config ?? SearchConfig.Play, seed);
        _states = new TGame[Batch];
        _leaves = new int[Batch];
        _logits = new float[Batch * TGame.ActionCount];
        _values = new float[Batch];
        Name = name ?? $"mcts{simulations}";
    }

    public void NewGame() => _tree.Reset(TGame.Initial);

    /// <summary>Points the tree at <paramref name="state"/>, reusing the existing search when possible.</summary>
    void SetRoot(in TGame state)
    {
        var root = _tree.RootState;
        if (ReuseTree)
        {
            if (root.Hash == state.Hash && root.Ply == state.Ply) return;
            if ((state.Ply == root.Ply + 1 || state.Ply == root.Ply + 2) && TryAdvanceTo(state)) return;
        }
        _tree.Reset(state);
    }

    bool TryAdvanceTo(in TGame target)
    {
        Span<int> moves = stackalloc int[TGame.ActionCount];
        Span<int> moves2 = stackalloc int[TGame.ActionCount];
        var root = _tree.RootState;
        int n = root.LegalMoves(moves);
        for (int i = 0; i < n; i++)
        {
            var a = root;
            a.Play(moves[i]);
            if (a.Ply == target.Ply && a.Hash == target.Hash)
            {
                _tree.Advance(moves[i]);
                return true;
            }
            if (target.Ply == a.Ply + 1)
            {
                int m = a.LegalMoves(moves2);
                for (int j = 0; j < m; j++)
                {
                    var b = a;
                    b.Play(moves2[j]);
                    if (b.Hash == target.Hash)
                    {
                        _tree.Advance(moves[i]);
                        _tree.Advance(moves2[j]);
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /// <summary>
    /// Runs simulations from <paramref name="state"/> until the root has <paramref name="simulations"/>
    /// visits (or cancellation). <paramref name="progress"/> is called after every batch.
    /// </summary>
    public SearchResult Search(in TGame state, int simulations, CancellationToken cancel = default,
                               Action<SearchTree<TGame>>? progress = null)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        SetRoot(state);
        int start = _tree.RootVisits;
        int a = TGame.ActionCount;
        bool noiseAdded = false;
        while (_tree.RootVisits < Math.Max(simulations, 1) || !_tree.RootExpanded)
        {
            if (cancel.IsCancellationRequested && _tree.RootExpanded) break;
            int want = Math.Min(Batch, Math.Max(1, simulations - _tree.RootVisits));
            int count = 0;
            for (int k = 0; k < want; k++)
            {
                var kind = _tree.SelectLeaf(out int leaf, out var s);
                if (kind == LeafKind.Collision) break;
                if (kind == LeafKind.NeedsEvaluation)
                {
                    _states[count] = s;
                    _leaves[count++] = leaf;
                }
            }
            if (count > 0)
            {
                _eval.Evaluate(_states.AsSpan(0, count), _logits, _values);
                for (int i = 0; i < count; i++)
                    _tree.Expand(_leaves[i], _states[i], _logits.AsSpan(i * a, a), _values[i]);
            }
            if (!noiseAdded && _tree.RootExpanded)
            {
                _tree.AddRootNoise();
                noiseAdded = true;
            }
            progress?.Invoke(_tree);
        }
        int best = _tree.ChooseMove(Temperature);
        LastResult = new SearchResult(best, _tree.RootValue, _tree.RootVisits - start, _tree.RootStats(),
                                      _tree.PrincipalVariation(), sw.Elapsed.TotalSeconds);
        return LastResult;
    }

    public int ChooseMove(in TGame state) => Search(state, Simulations).BestMove;
}

/// <summary>A perfect Connect Four player: always plays a move with the best exact score.</summary>
public sealed class SolverPlayer(int seed = 0) : IPlayer<ConnectFour>
{
    readonly Solver.ConnectFourSolver _solver = new();
    readonly Random _rng = new(seed);
    public string Name => "perfect";

    public int ChooseMove(in ConnectFour state)
    {
        var scores = _solver.ScoreMoves(state);
        int best = scores.Where(s => s.HasValue).Max(s => s!.Value);
        var options = Enumerable.Range(0, scores.Length).Where(c => scores[c] == best).ToArray();
        return options[_rng.Next(options.Length)];
    }
}
