using System.Globalization;
using Gambit.Core.Games;
using Gambit.Core.Search;

namespace Gambit.Core.Solver;

/// <summary>A Connect Four position with the exact score of every move.</summary>
public sealed record SolvedPosition(string Moves, int?[] Scores)
{
    public ConnectFour State => ConnectFour.FromMoves(Moves);
    public int Best => Scores.Where(s => s.HasValue).Max(s => s!.Value);
    public bool IsOptimal(int move) => Scores[move] == Best;
    /// <summary>The game-theoretic result for the side to move: 1 win, 0 draw, -1 loss.</summary>
    public int Result => Math.Sign(Best);
    /// <summary>A move that keeps the theoretical result (wins stay wins, draws stay draws).</summary>
    public bool KeepsResult(int move) => Scores[move].HasValue && Math.Sign(Scores[move]!.Value) == Result;
}

/// <summary>Accuracy of a player against the perfect solver.</summary>
/// <param name="Optimal">Share of positions where the move has the best exact score (fastest win / slowest loss).</param>
/// <param name="KeepsResult">Share of positions where the move keeps the theoretical result.</param>
/// <param name="BlunderRate">Among positions that are not already lost, the share where the move throws
/// away the win or the draw - the measure that matters for a player that, like AlphaZero, maximises
/// its chance of winning rather than the speed of the win.</param>
/// <param name="ValueSign">Share of positions where the value's sign (win / draw / loss) is right.</param>
public sealed record SolverAccuracy(int Positions, double Optimal, double KeepsResult, double BlunderRate, double ValueSign)
{
    public override string ToString() =>
        $"blunders {100 * BlunderRate:F1}%, result-preserving {100 * KeepsResult:F1}%, fastest-optimal {100 * Optimal:F1}%, " +
        $"value sign {100 * ValueSign:F1}% ({Positions} positions)";
}

/// <summary>
/// A test set of Connect Four positions solved exactly, to grade the AI against perfect play.
///
/// Positions come from games with random moves, 10 to 30 moves deep (shallower positions take the
/// solver seconds each). Only real decisions are kept: no immediate win is available, there is no
/// opponent threat to block (those one-move tactics are easy for any search), and at least one
/// legal move must be worse than the best. Two measures are reported: the share of positions where the AI
/// plays a move with the best score ("optimal"), and the share where its move at least preserves
/// the theoretical result (a win stays a win, possibly a slower one; a draw stays a draw).
/// </summary>
public static class SolvedPositions
{
    public static List<SolvedPosition> Generate(int count, int minPly, int maxPly, int seed, Action<int>? progress = null)
    {
        var result = new List<SolvedPosition>(count);
        var seen = new HashSet<ulong>();
        object gate = new();
        int produced = 0;
        Parallel.For(0, Environment.ProcessorCount, worker =>
        {
            var solver = new ConnectFourSolver(22);
            var rng = new Random(seed * 1000 + worker);
            Span<int> moves = stackalloc int[7];
            while (true)
            {
                lock (gate)
                    if (produced >= count) return;
                int ply = rng.Next(minPly, maxPly + 1);
                var g = ConnectFour.Initial;
                var text = new System.Text.StringBuilder();
                while (g.Ply < ply && !g.Outcome.IsOver())
                {
                    int n = g.LegalMoves(moves);
                    int m = moves[rng.Next(n)];
                    g.Play(m);
                    text.Append((char)('1' + m));
                }
                if (g.Outcome.IsOver() || HasTactic(g)) continue;
                lock (gate)
                    if (!seen.Add(g.Hash)) continue;
                var scores = solver.ScoreMoves(g);
                var legal = scores.Where(s => s.HasValue).Select(s => s!.Value).ToList();
                if (legal.Distinct().Count() < 2) continue;   // every move equally good: no decision to grade
                lock (gate)
                {
                    if (produced >= count) return;
                    result.Add(new SolvedPosition(text.ToString(), scores));
                    produced++;
                    progress?.Invoke(produced);
                }
            }
        });
        return result.OrderBy(p => p.Moves.Length).ThenBy(p => p.Moves, StringComparer.Ordinal).ToList();
    }

    static bool HasTactic(in ConnectFour g)
    {
        for (int c = 0; c < 7; c++)
            if (g.IsLegal(c) && (g.IsWinningMove(c) || g.IsOpponentWinningMove(c))) return true;
        return false;
    }

    /// <summary>One line per position: the moves, then the seven scores ("-" for a full column).</summary>
    public static void Save(string path, IEnumerable<SolvedPosition> positions) =>
        File.WriteAllLines(path, positions.Select(p =>
            p.Moves + " " + string.Join(" ", p.Scores.Select(s => s?.ToString(CultureInfo.InvariantCulture) ?? "-"))));

    public static List<SolvedPosition> Load(string path) =>
        File.ReadLines(path).Where(l => l.Length > 0).Select(l =>
        {
            var f = l.Split(' ');
            return new SolvedPosition(f[0], f.Skip(1).Select(x => x == "-" ? (int?)null : int.Parse(x, CultureInfo.InvariantCulture)).ToArray());
        }).ToList();

    /// <summary>
    /// Grades an evaluator: with <paramref name="simulations"/> = 0 the raw network (policy argmax),
    /// otherwise MCTS with that many simulations per position. Positions are searched in parallel
    /// in batches, so a GPU evaluator gets full batches.
    /// </summary>
    public static SolverAccuracy Grade(IReadOnlyList<SolvedPosition> set, IEvaluator<ConnectFour> eval, int simulations, int parallel = 256)
    {
        int optimal = 0, keeps = 0, sign = 0, notLost = 0, blunders = 0;
        var states = set.Select(p => p.State).ToArray();
        var chosen = new int[set.Count];
        var value = new float[set.Count];
        if (simulations <= 0)
        {
            var logits = new float[states.Length * 7];
            var values = new float[states.Length];
            for (int start = 0; start < states.Length; start += parallel)
            {
                int n = Math.Min(parallel, states.Length - start);
                eval.Evaluate(states.AsSpan(start, n), logits.AsSpan(start * 7), values.AsSpan(start));
            }
            for (int i = 0; i < states.Length; i++)
            {
                int best = -1;
                for (int c = 0; c < 7; c++)
                    if (states[i].IsLegal(c) && (best < 0 || logits[i * 7 + c] > logits[i * 7 + best])) best = c;
                chosen[i] = best;
                value[i] = values[i];
            }
        }
        else
        {
            var results = BatchedSearch.Run(states, eval, simulations, parallel);
            for (int i = 0; i < states.Length; i++)
            {
                chosen[i] = results[i].Move;
                value[i] = results[i].Value;
            }
        }
        for (int i = 0; i < set.Count; i++)
        {
            if (set[i].IsOptimal(chosen[i])) optimal++;
            if (set[i].KeepsResult(chosen[i])) keeps++;
            if (set[i].Result >= 0)
            {
                notLost++;
                if (!set[i].KeepsResult(chosen[i])) blunders++;
            }
            int predicted = value[i] > 0.33f ? 1 : value[i] < -0.33f ? -1 : 0;
            if (predicted == set[i].Result) sign++;
        }
        double n0 = Math.Max(set.Count, 1);
        return new SolverAccuracy(set.Count, optimal / n0, keeps / n0, blunders / (double)Math.Max(notLost, 1), sign / n0);
    }
}

/// <summary>Searches many independent positions at once, batching their leaves.</summary>
public static class BatchedSearch
{
    public static (int Move, float Value)[] Run<TGame>(TGame[] positions, IEvaluator<TGame> eval, int simulations, int parallel = 256)
        where TGame : struct, IGame<TGame>
    {
        var output = new (int, float)[positions.Length];
        int a = TGame.ActionCount;
        for (int start = 0; start < positions.Length; start += parallel)
        {
            int n = Math.Min(parallel, positions.Length - start);
            var trees = new SearchTree<TGame>[n];
            for (int i = 0; i < n; i++)
            {
                trees[i] = new SearchTree<TGame>(SearchConfig.Play, i);
                trees[i].Reset(positions[start + i]);
            }
            var leaves = new int[n];
            var states = new TGame[n];
            var slot = new int[n];
            var logits = new float[n * a];
            var values = new float[n];
            while (true)
            {
                int count = 0;
                for (int i = 0; i < n; i++)
                {
                    if (trees[i].RootExpanded && trees[i].RootVisits >= simulations) continue;
                    if (trees[i].SelectLeaf(out int leaf, out var s) == LeafKind.NeedsEvaluation)
                    {
                        leaves[count] = leaf;
                        states[count] = s;
                        slot[count++] = i;
                    }
                }
                if (count == 0 && trees.All(t => t.RootExpanded && t.RootVisits >= simulations)) break;
                if (count == 0) continue;
                eval.Evaluate(states.AsSpan(0, count), logits, values);
                for (int k = 0; k < count; k++)
                    trees[slot[k]].Expand(leaves[k], states[k], logits.AsSpan(k * a, a), values[k]);
            }
            for (int i = 0; i < n; i++) output[start + i] = (trees[i].ChooseMove(0), trees[i].RootValue);
        }
        return output;
    }
}
