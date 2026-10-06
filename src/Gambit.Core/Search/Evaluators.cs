using Gambit.Core.Games;

namespace Gambit.Core.Search;

/// <summary>
/// Something that scores positions for the search: policy logits over all actions and a value in
/// [-1, 1] for the side to move. Positions arrive in batches, so a network can evaluate many leaves
/// (from many games, or several leaves of one tree) in one GPU call.
/// </summary>
public interface IEvaluator<TGame> where TGame : struct, IGame<TGame>
{
    /// <summary>
    /// Evaluates <paramref name="states"/>: writes states.Length x ActionCount logits into
    /// <paramref name="logits"/> and one value per state into <paramref name="values"/>.
    /// Implementations must be safe to call from one thread at a time.
    /// </summary>
    void Evaluate(ReadOnlySpan<TGame> states, Span<float> logits, Span<float> values);
}

/// <summary>Uniform policy, value 0: the search alone (a pure-MCTS baseline without rollouts).</summary>
public sealed class UniformEvaluator<TGame> : IEvaluator<TGame> where TGame : struct, IGame<TGame>
{
    public void Evaluate(ReadOnlySpan<TGame> states, Span<float> logits, Span<float> values)
    {
        logits[..(states.Length * TGame.ActionCount)].Clear();
        values[..states.Length].Clear();
    }
}

/// <summary>
/// Classic Monte Carlo evaluation: uniform policy, value = average result of random playouts.
/// With MCTS on top this is the pre-AlphaZero baseline ("pure MCTS"), a useful fixed anchor for Elo.
/// </summary>
public sealed class RolloutEvaluator<TGame> : IEvaluator<TGame> where TGame : struct, IGame<TGame>
{
    readonly int _rollouts;
    readonly Random _rng;

    public RolloutEvaluator(int rollouts = 1, int seed = 0)
    {
        _rollouts = rollouts;
        _rng = new Random(seed);
    }

    public void Evaluate(ReadOnlySpan<TGame> states, Span<float> logits, Span<float> values)
    {
        logits[..(states.Length * TGame.ActionCount)].Clear();
        Span<int> moves = stackalloc int[TGame.ActionCount];
        for (int i = 0; i < states.Length; i++)
        {
            float total = 0;
            for (int r = 0; r < _rollouts; r++)
            {
                var g = states[i];
                int side = g.SideToMove;
                while (!g.Outcome.IsOver())
                {
                    int n = g.LegalMoves(moves);
                    g.Play(moves[_rng.Next(n)]);
                }
                // Loss means the side to move at the end lost, i.e. the last mover won.
                if (g.Outcome == Outcome.Loss) total += g.SideToMove == side ? -1f : 1f;
            }
            values[i] = total / _rollouts;
        }
    }
}

/// <summary>
/// Caches evaluations by position hash in front of another evaluator. Self-play games all start
/// from the same position and share many openings, and searches revisit transpositions, so a
/// sizeable fraction of network calls are repeats. The table is lock-free: each slot holds an
/// immutable entry swapped in atomically, and a hit must match the full hash.
/// </summary>
public sealed class CachedEvaluator<TGame> : IEvaluator<TGame> where TGame : struct, IGame<TGame>
{
    sealed class Entry(ulong hash, float[] logits, float value)
    {
        public readonly ulong Hash = hash;
        public readonly float[] Logits = logits;
        public readonly float Value = value;
    }

    readonly IEvaluator<TGame> _inner;
    readonly Entry?[] _table;
    readonly int _mask;
    TGame[] _missStates = new TGame[64];
    int[] _missIndex = new int[64];
    float[] _missLogits = new float[64 * TGame.ActionCount];
    float[] _missValues = new float[64];

    public long Hits, Misses;

    public CachedEvaluator(IEvaluator<TGame> inner, int sizeLog2 = 20)
    {
        _inner = inner;
        _table = new Entry?[1 << sizeLog2];
        _mask = (1 << sizeLog2) - 1;
    }

    public IEvaluator<TGame> Inner => _inner;
    public double HitRate => Hits + Misses == 0 ? 0 : (double)Hits / (Hits + Misses);

    public void Clear() => Array.Clear(_table);

    public void Evaluate(ReadOnlySpan<TGame> states, Span<float> logits, Span<float> values)
    {
        int a = TGame.ActionCount, misses = 0;
        if (_missStates.Length < states.Length)
        {
            _missStates = new TGame[states.Length];
            _missIndex = new int[states.Length];
            _missLogits = new float[states.Length * a];
            _missValues = new float[states.Length];
        }
        for (int i = 0; i < states.Length; i++)
        {
            ulong h = states[i].Hash;
            var e = Volatile.Read(ref _table[(int)h & _mask]);
            if (e != null && e.Hash == h)
            {
                e.Logits.CopyTo(logits.Slice(i * a, a));
                values[i] = e.Value;
                Hits++;
            }
            else
            {
                _missStates[misses] = states[i];
                _missIndex[misses++] = i;
            }
        }
        Misses += misses;
        if (misses == 0) return;
        _inner.Evaluate(_missStates.AsSpan(0, misses), _missLogits, _missValues);
        for (int k = 0; k < misses; k++)
        {
            int i = _missIndex[k];
            var l = _missLogits.AsSpan(k * a, a);
            l.CopyTo(logits.Slice(i * a, a));
            values[i] = _missValues[k];
            ulong h = states[i].Hash;
            Volatile.Write(ref _table[(int)h & _mask], new Entry(h, l.ToArray(), _missValues[k]));
        }
    }
}
