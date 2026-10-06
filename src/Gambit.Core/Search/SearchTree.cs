using System.Runtime.CompilerServices;
using Gambit.Core.Games;

namespace Gambit.Core.Search;

/// <summary>Search parameters (AlphaZero's PUCT with a few well-known refinements).</summary>
public sealed record SearchConfig
{
    /// <summary>Exploration constant: c(N) = CpuctInit + CpuctFactor * log((N + CpuctBase + 1) / CpuctBase).</summary>
    public float CpuctInit { get; init; } = 1.25f;
    public float CpuctBase { get; init; } = 19652f;
    public float CpuctFactor { get; init; } = 1f;

    /// <summary>First-play urgency: an unvisited child is valued at the parent's value minus this
    /// times sqrt(prior mass already visited) - so the search explores siblings of good moves
    /// before it widens to everything.</summary>
    public float FpuReduction { get; init; } = 0.25f;

    /// <summary>Root Dirichlet noise for self-play exploration (0 disables).</summary>
    public float DirichletAlpha { get; init; } = 1.0f;
    public float NoiseFraction { get; init; } = 0f;

    /// <summary>Softmax temperature applied to the network's policy logits at expansion.</summary>
    public float PolicyTemperature { get; init; } = 1f;

    public static SearchConfig Play => new();
    public static SearchConfig SelfPlay(float alpha) => new() { NoiseFraction = 0.25f, DirichletAlpha = alpha };
}

/// <summary>A tree node. Children of a node are stored contiguously from <see cref="FirstChild"/>.</summary>
public struct Node
{
    public int Parent;
    public int FirstChild;      // -1 until expanded
    public float Prior;
    public float ValueSum;      // sum of values from the point of view of the player who moved INTO this node
    public int Visits;
    public short Virtual;       // pending evaluations passing through this node (virtual loss)
    public short Action;        // the move that leads here from the parent
    public byte ChildCount;
    public NodeFlags Flags;
}

[Flags]
public enum NodeFlags : byte
{
    None = 0,
    Expanded = 1,
    Pending = 2,          // waiting for a network evaluation
    Terminal = 4,
    TerminalDraw = 8,     // with Terminal: a draw; without: the side to move here has lost
}

public enum LeafKind { NeedsEvaluation, Terminal, Collision }

/// <summary>Statistics of one root move, for display and for building training targets.</summary>
public readonly record struct MoveStats(int Action, int Visits, float Prior, float Q);

/// <summary>
/// The Monte Carlo search tree for one game, PUCT selection as in AlphaZero.
///
/// The search is split into steps so that the caller can batch network evaluations - across many
/// games in self-play, and across several leaves of this tree (with virtual loss) when playing
/// one game:
///     SelectLeaf  walks from the root to a leaf (applying virtual loss along the path);
///     Expand      creates the leaf's children from the network's policy and backs up its value.
/// Nodes live in one array of structs, children contiguous; nothing is allocated per simulation.
/// When a move is played the chosen subtree is kept and compacted (<see cref="Advance"/>).
/// </summary>
public sealed class SearchTree<TGame> where TGame : struct, IGame<TGame>
{
    Node[] _nodes;
    Node[] _spare;
    int _count;
    readonly SearchConfig _cfg;
    readonly Random _rng;
    readonly float[] _scratch = new float[TGame.ActionCount];
    readonly int[] _moves = new int[TGame.ActionCount];

    public TGame RootState { get; private set; }
    public const int Root = 0;
    public SearchConfig Config => _cfg;

    public SearchTree(SearchConfig cfg, int seed, int capacity = 4096)
    {
        _cfg = cfg;
        _rng = new Random(seed);
        _nodes = new Node[capacity];
        _spare = new Node[capacity];
        Reset(TGame.Initial);
    }

    public int NodeCount => _count;
    public ref Node this[int i] => ref _nodes[i];
    public ref Node RootNode => ref _nodes[Root];
    public bool RootExpanded => (_nodes[Root].Flags & NodeFlags.Expanded) != 0;
    public int RootVisits => _nodes[Root].Visits;

    public void Reset(in TGame state)
    {
        RootState = state;
        _count = 1;
        _nodes[Root] = new Node { Parent = -1, FirstChild = -1 };
    }

    int Allocate(int n)
    {
        if (_count + n > _nodes.Length)
        {
            int size = Math.Max(_nodes.Length * 2, _count + n);
            Array.Resize(ref _nodes, size);
        }
        int first = _count;
        _count += n;
        return first;
    }

    float Cpuct(int parentVisits) =>
        _cfg.CpuctInit + _cfg.CpuctFactor * MathF.Log((parentVisits + _cfg.CpuctBase + 1f) / _cfg.CpuctBase);

    /// <summary>Index of the child to descend into from <paramref name="parent"/>.</summary>
    int SelectChild(int parent)
    {
        ref Node p = ref _nodes[parent];
        int first = p.FirstChild, n = p.ChildCount;
        int parentN = p.Visits + p.Virtual;
        float sqrtN = MathF.Sqrt(Math.Max(parentN, 1));
        float c = Cpuct(parentN) * sqrtN;

        // Value of the position for the player to move at the parent (its ValueSum is from the
        // other side's point of view), used as the basis for unvisited children.
        float parentQ = p.Visits > 0 ? -p.ValueSum / p.Visits : 0f;
        float visitedPrior = 0f;
        for (int i = 0; i < n; i++)
            if (_nodes[first + i].Visits + _nodes[first + i].Virtual > 0) visitedPrior += _nodes[first + i].Prior;
        float fpu = parentQ - _cfg.FpuReduction * MathF.Sqrt(visitedPrior);

        int best = first;
        float bestScore = float.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            ref Node ch = ref _nodes[first + i];
            int visits = ch.Visits + ch.Virtual;
            // Virtual loss: each pending visit counts as a loss for the mover.
            float q = visits > 0 ? (ch.ValueSum - ch.Virtual) / visits : fpu;
            float score = q + c * ch.Prior / (1 + visits);
            if (score > bestScore)
            {
                bestScore = score;
                best = first + i;
            }
        }
        return best;
    }

    /// <summary>
    /// Walks to a leaf. On NeedsEvaluation the leaf is marked pending and virtual loss is applied
    /// along the path; the caller must evaluate <paramref name="state"/> and call Expand. Terminal
    /// leaves are backed up immediately. A Collision means the walk reached a leaf that is already
    /// waiting for an evaluation (nothing changed).
    /// </summary>
    public LeafKind SelectLeaf(out int leaf, out TGame state)
    {
        state = RootState;
        int node = Root;
        while ((_nodes[node].Flags & NodeFlags.Expanded) != 0)
        {
            node = SelectChild(node);
            state.Play(_nodes[node].Action);
        }
        leaf = node;
        ref Node l = ref _nodes[node];
        if ((l.Flags & NodeFlags.Terminal) == 0 && state.Outcome.IsOver())
        {
            l.Flags |= NodeFlags.Terminal;
            if (state.Outcome == Outcome.Draw) l.Flags |= NodeFlags.TerminalDraw;
        }
        if ((l.Flags & NodeFlags.Terminal) != 0)
        {
            Backup(node, (l.Flags & NodeFlags.TerminalDraw) != 0 ? 0f : -1f);
            return LeafKind.Terminal;
        }
        if ((l.Flags & NodeFlags.Pending) != 0) return LeafKind.Collision;
        l.Flags |= NodeFlags.Pending;
        for (int n = node; n >= 0; n = _nodes[n].Parent) _nodes[n].Virtual++;
        return LeafKind.NeedsEvaluation;
    }

    /// <summary>
    /// Expands a pending leaf with the network's output for <paramref name="state"/>: policy logits
    /// (ActionCount entries) and a value in [-1, 1] for the side to move, then backs the value up.
    /// </summary>
    public void Expand(int leaf, in TGame state, ReadOnlySpan<float> logits, float value)
    {
        int count = state.LegalMoves(_moves);
        int first = Allocate(count);
        ref Node l = ref _nodes[leaf];
        // Softmax over the legal moves only.
        float invT = 1f / _cfg.PolicyTemperature;
        float max = float.NegativeInfinity;
        for (int i = 0; i < count; i++) max = MathF.Max(max, logits[_moves[i]] * invT);
        float sum = 0f;
        for (int i = 0; i < count; i++)
        {
            float e = MathF.Exp(logits[_moves[i]] * invT - max);
            _scratch[i] = e;
            sum += e;
        }
        for (int i = 0; i < count; i++)
        {
            _nodes[first + i] = new Node
            {
                Parent = leaf, FirstChild = -1, Prior = _scratch[i] / sum, Action = (short)_moves[i],
            };
        }
        l.FirstChild = first;
        l.ChildCount = (byte)count;
        l.Flags = (l.Flags & ~NodeFlags.Pending) | NodeFlags.Expanded;
        for (int n = leaf; n >= 0; n = _nodes[n].Parent) _nodes[n].Virtual--;
        Backup(leaf, value);
    }

    /// <summary>Adds a visit with value <paramref name="value"/> (for the side to move at <paramref name="node"/>) up to the root.</summary>
    void Backup(int node, float value)
    {
        // The node's ValueSum is from the point of view of the player who moved into it: the opponent.
        float v = -value;
        for (int n = node; n >= 0; n = _nodes[n].Parent)
        {
            _nodes[n].Visits++;
            _nodes[n].ValueSum += v;
            v = -v;
        }
    }

    /// <summary>Mixes Dirichlet(alpha) noise into the root priors (self-play exploration).</summary>
    public void AddRootNoise()
    {
        if (_cfg.NoiseFraction <= 0 || !RootExpanded) return;
        ref Node r = ref _nodes[Root];
        int n = r.ChildCount;
        Span<float> noise = stackalloc float[n];
        float sum = 0;
        for (int i = 0; i < n; i++) sum += noise[i] = Gamma(_cfg.DirichletAlpha);
        float f = _cfg.NoiseFraction;
        for (int i = 0; i < n; i++)
        {
            ref Node ch = ref _nodes[r.FirstChild + i];
            ch.Prior = (1 - f) * ch.Prior + f * noise[i] / Math.Max(sum, 1e-20f);
        }
    }

    /// <summary>Gamma(alpha, 1) sample (Marsaglia-Tsang; boosted for alpha < 1).</summary>
    float Gamma(float alpha)
    {
        if (alpha < 1f)
            return Gamma(alpha + 1f) * MathF.Pow((float)_rng.NextDouble(), 1f / alpha);
        float d = alpha - 1f / 3f, c = 1f / MathF.Sqrt(9f * d);
        while (true)
        {
            float x, v;
            do
            {
                x = Gaussian();
                v = 1f + c * x;
            } while (v <= 0f);
            v = v * v * v;
            float u = (float)_rng.NextDouble();
            if (u < 1f - 0.0331f * x * x * x * x) return d * v;
            if (MathF.Log(u) < 0.5f * x * x + d * (1f - v + MathF.Log(v))) return d * v;
        }
    }

    float Gaussian()
    {
        double u1 = 1.0 - _rng.NextDouble(), u2 = _rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
    }

    /// <summary>Visits, prior and value of every root move.</summary>
    public MoveStats[] RootStats()
    {
        ref Node r = ref _nodes[Root];
        if ((r.Flags & NodeFlags.Expanded) == 0) return [];
        var stats = new MoveStats[r.ChildCount];
        for (int i = 0; i < r.ChildCount; i++)
        {
            ref Node ch = ref _nodes[r.FirstChild + i];
            stats[i] = new MoveStats(ch.Action, ch.Visits, ch.Prior, ch.Visits > 0 ? ch.ValueSum / ch.Visits : float.NaN);
        }
        return stats;
    }

    /// <summary>The root value for the side to move, in [-1, 1].</summary>
    public float RootValue => _nodes[Root].Visits > 0 ? -_nodes[Root].ValueSum / _nodes[Root].Visits : 0f;

    /// <summary>Writes the root visit distribution (the policy training target) into <paramref name="target"/>.</summary>
    public void VisitDistribution(Span<float> target)
    {
        target[..TGame.ActionCount].Clear();
        ref Node r = ref _nodes[Root];
        float total = 0;
        for (int i = 0; i < r.ChildCount; i++) total += _nodes[r.FirstChild + i].Visits;
        if (total <= 0) return;
        for (int i = 0; i < r.ChildCount; i++)
        {
            ref Node ch = ref _nodes[r.FirstChild + i];
            target[ch.Action] = ch.Visits / total;
        }
    }

    /// <summary>
    /// Picks a root move: proportionally to visits^(1/temperature), or the most visited (ties
    /// broken by value) when temperature is 0.
    /// </summary>
    public int ChooseMove(float temperature)
    {
        ref Node r = ref _nodes[Root];
        if ((r.Flags & NodeFlags.Expanded) == 0) throw new InvalidOperationException("search the position first");
        int first = r.FirstChild, n = r.ChildCount;
        if (temperature <= 1e-3f)
        {
            int best = first;
            for (int i = 1; i < n; i++)
            {
                ref Node a = ref _nodes[first + i];
                ref Node b = ref _nodes[best];
                if (a.Visits > b.Visits ||
                    (a.Visits == b.Visits && a.Visits > 0 && a.ValueSum / a.Visits > b.ValueSum / b.Visits))
                    best = first + i;
            }
            return _nodes[best].Action;
        }
        Span<double> w = stackalloc double[n];
        double total = 0;
        for (int i = 0; i < n; i++) total += w[i] = Math.Pow(_nodes[first + i].Visits, 1.0 / temperature);
        if (total <= 0) return _nodes[first + _rng.Next(n)].Action;
        double x = _rng.NextDouble() * total;
        for (int i = 0; i < n; i++)
        {
            x -= w[i];
            if (x <= 0) return _nodes[first + i].Action;
        }
        return _nodes[first + n - 1].Action;
    }

    /// <summary>
    /// Plays <paramref name="action"/> at the root and keeps its subtree as the new tree, copying it
    /// to the front of a fresh node array (so the arrays never grow beyond one move's search).
    /// </summary>
    public void Advance(int action)
    {
        var next = RootState;
        next.Play(action);
        int child = -1;
        ref Node r = ref _nodes[Root];
        if ((r.Flags & NodeFlags.Expanded) != 0)
            for (int i = 0; i < r.ChildCount; i++)
                if (_nodes[r.FirstChild + i].Action == action) { child = r.FirstChild + i; break; }
        if (child < 0 || (_nodes[child].Flags & NodeFlags.Pending) != 0 || HasPending(child))
        {
            Reset(next);
            return;
        }
        // Breadth-first copy of the subtree; children stay contiguous.
        if (_spare.Length < _count) _spare = new Node[_nodes.Length];
        int count = 1;
        _spare[0] = _nodes[child];
        _spare[0].Parent = -1;
        for (int i = 0; i < count; i++)
        {
            ref Node src = ref _spare[i];
            if ((src.Flags & NodeFlags.Expanded) == 0) continue;
            int oldFirst = src.FirstChild;
            src.FirstChild = count;
            for (int k = 0; k < src.ChildCount; k++)
            {
                _spare[count + k] = _nodes[oldFirst + k];
                _spare[count + k].Parent = i;
            }
            count += src.ChildCount;
        }
        (_nodes, _spare) = (_spare, _nodes);
        _count = count;
        RootState = next;
    }

    bool HasPending(int node) => _nodes[node].Virtual != 0;

    /// <summary>The most visited line from the root.</summary>
    public List<int> PrincipalVariation(int maxLength = 12)
    {
        var pv = new List<int>();
        int node = Root;
        while (pv.Count < maxLength && (_nodes[node].Flags & NodeFlags.Expanded) != 0)
        {
            ref Node p = ref _nodes[node];
            int best = -1;
            for (int i = 0; i < p.ChildCount; i++)
                if (best < 0 || _nodes[p.FirstChild + i].Visits > _nodes[best].Visits) best = p.FirstChild + i;
            if (best < 0 || _nodes[best].Visits == 0) break;
            pv.Add(_nodes[best].Action);
            node = best;
        }
        return pv;
    }

    /// <summary>Index of the root child for <paramref name="action"/>, or -1.</summary>
    public int ChildOf(int node, int action)
    {
        ref Node p = ref _nodes[node];
        for (int i = 0; i < p.ChildCount; i++)
            if (_nodes[p.FirstChild + i].Action == action) return p.FirstChild + i;
        return -1;
    }

    /// <summary>Undoes a pending selection whose evaluation will not arrive (e.g. a cancelled batch).</summary>
    public void CancelPending(int leaf)
    {
        if ((_nodes[leaf].Flags & NodeFlags.Pending) == 0) return;
        _nodes[leaf].Flags &= ~NodeFlags.Pending;
        for (int n = leaf; n >= 0; n = _nodes[n].Parent) _nodes[n].Virtual--;
    }

    internal Random Rng => _rng;
}
