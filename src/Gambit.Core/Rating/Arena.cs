using Gambit.Core.Games;
using Gambit.Core.Search;

namespace Gambit.Core.Rating;

/// <summary>Result of a match, from player A's point of view.</summary>
public readonly record struct MatchResult(int WinsA, int Draws, int WinsB)
{
    public int Games => WinsA + Draws + WinsB;
    public double ScoreA => Games == 0 ? 0.5 : (WinsA + 0.5 * Draws) / Games;
    public double EloDifference => EloTable.DifferenceFromScore(ScoreA);

    /// <summary>
    /// Likelihood of superiority: the probability that A is stronger than B given the decisive games
    /// (draws carry no information about which is stronger), Phi((wins - losses) / sqrt(wins + losses)).
    /// The usual test in engine development; unlike a fixed score threshold it stays meaningful when
    /// most games are drawn.
    /// </summary>
    public double LikelihoodOfSuperiority => WinsA + WinsB == 0 ? 0.5 : NormalCdf((WinsA - WinsB) / Math.Sqrt(WinsA + WinsB));

    static double NormalCdf(double x)
    {
        // Abramowitz-Stegun 7.1.26 approximation of erf, accurate to 1.5e-7.
        double t = 1 / (1 + 0.3275911 * Math.Abs(x) / Math.Sqrt(2));
        double y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
                   * Math.Exp(-x * x / 2);
        return x >= 0 ? (1 + y) / 2 : (1 - y) / 2;
    }
    public override string ToString() =>
        $"+{WinsA} ={Draws} -{WinsB} ({100 * ScoreA:F1}%, {EloDifference:+0;-0} Elo, LOS {100 * LikelihoodOfSuperiority:F0}%)";
}

public sealed record ArenaConfig
{
    public int Games { get; init; } = 200;
    public int SimulationsA { get; init; } = 200;
    public int SimulationsB { get; init; } = 200;
    /// <summary>Games played at once (their leaves are batched per evaluator).</summary>
    public int Parallel { get; init; } = 100;
    /// <summary>Each game starts with this many uniformly random moves, so the games differ; games 2k
    /// and 2k+1 share their opening with colours swapped, which cancels out an unbalanced opening.</summary>
    public int RandomOpeningMoves { get; init; } = 4;
    /// <summary>Leaves each game selects per step (with virtual loss): more means fewer, larger batches.</summary>
    public int LeavesPerStepA { get; init; } = 1;
    public int LeavesPerStepB { get; init; } = 1;
    public int Seed { get; init; } = 99;
}

/// <summary>
/// Matches between two searchers. Each game has two trees (one per player, both kept up to date
/// with every move); at each step every game's player-to-move selects a leaf, and the leaves are
/// evaluated in two batches, one per evaluator. Colours alternate: A plays first in even games.
/// </summary>
public static class Arena
{
    public static MatchResult Play<TGame>(IEvaluator<TGame> a, IEvaluator<TGame> b, ArenaConfig cfg,
                                          Action<MatchResult>? progress = null, CancellationToken cancel = default)
        where TGame : struct, IGame<TGame>
    {
        var search = SearchConfig.Play;
        int parallel = Math.Min(cfg.Parallel, cfg.Games);
        var trees = new SearchTree<TGame>[parallel, 2];   // [slot, player 0 = A, 1 = B]
        var aFirst = new bool[parallel];
        var active = new bool[parallel];
        int nextGame = 0, winsA = 0, draws = 0, winsB = 0, done = 0;
        // The random opening of game g (shared by games 2k and 2k+1).
        TGame Opening(int game)
        {
            var rng = new Random(cfg.Seed * 7919 + game / 2);
            Span<int> moves = stackalloc int[TGame.ActionCount];
            while (true)
            {
                var s = TGame.Initial;
                for (int k = 0; k < cfg.RandomOpeningMoves && !s.Outcome.IsOver(); k++)
                {
                    int n = s.LegalMoves(moves);
                    s.Play(moves[rng.Next(n)]);
                }
                if (!s.Outcome.IsOver()) return s;
            }
        }

        for (int i = 0; i < parallel; i++)
        {
            trees[i, 0] = new SearchTree<TGame>(search, cfg.Seed * 31 + 2 * i);
            trees[i, 1] = new SearchTree<TGame>(search, cfg.Seed * 31 + 2 * i + 1);
            var start = Opening(nextGame);
            trees[i, 0].Reset(start);
            trees[i, 1].Reset(start);
            aFirst[i] = nextGame % 2 == 0;
            active[i] = true;
            nextGame++;
        }
        int act = TGame.ActionCount;
        var perStep = new[] { Math.Clamp(cfg.LeavesPerStepA, 1, 16), Math.Clamp(cfg.LeavesPerStepB, 1, 16) };
        int maxLeaves = Math.Max(perStep[0], perStep[1]);
        var leaves = new int[parallel, maxLeaves];
        var stage = new TGame[parallel, maxLeaves];
        var leafCount = new int[parallel];
        var states = new TGame[2][] { new TGame[parallel * maxLeaves], new TGame[parallel * maxLeaves] };
        var slotOf = new (int Slot, int Leaf)[2][] { new (int, int)[parallel * maxLeaves], new (int, int)[parallel * maxLeaves] };
        var counts = new int[2];
        var logits = new float[parallel * maxLeaves * act];
        var values = new float[parallel * maxLeaves];
        var evals = new[] { a, b };
        var sims = new[] { cfg.SimulationsA, cfg.SimulationsB };

        // Which player (0 = A, 1 = B) moves in slot i: A has the side that moves first after the opening
        // in even games, the other side in odd games.
        int openingParity = cfg.RandomOpeningMoves & 1;
        int Mover(int i) => ((trees[i, 0].RootState.SideToMove ^ openingParity) == 0) == aFirst[i] ? 0 : 1;

        while (done < cfg.Games && !cancel.IsCancellationRequested)
        {
            counts[0] = counts[1] = 0;
            System.Threading.Tasks.Parallel.For(0, parallel, i =>
            {
                leafCount[i] = 0;
                if (!active[i]) return;
                int p = Mover(i);
                var tree = trees[i, p];
                int want = tree.RootExpanded ? Math.Min(perStep[p], Math.Max(1, sims[p] - tree.RootVisits)) : 1;
                for (int k = 0; k < want; k++)
                {
                    var kind = tree.SelectLeaf(out int leaf, out var s);
                    if (kind == LeafKind.Collision) break;
                    if (kind == LeafKind.NeedsEvaluation)
                    {
                        leaves[i, leafCount[i]] = leaf;
                        stage[i, leafCount[i]++] = s;
                    }
                }
            });
            for (int i = 0; i < parallel; i++)
            {
                if (leafCount[i] == 0) continue;
                int p = Mover(i);
                for (int k = 0; k < leafCount[i]; k++)
                {
                    states[p][counts[p]] = stage[i, k];
                    slotOf[p][counts[p]++] = (i, k);
                }
            }
            for (int p = 0; p < 2; p++)
            {
                if (counts[p] == 0) continue;
                evals[p].Evaluate(states[p].AsSpan(0, counts[p]), logits, values);
                for (int k = 0; k < counts[p]; k++)
                {
                    var (i, l) = slotOf[p][k];
                    trees[i, p].Expand(leaves[i, l], states[p][k], logits.AsSpan(k * act, act), values[k]);
                }
            }
            for (int i = 0; i < parallel; i++)
            {
                if (!active[i]) continue;
                int p = Mover(i);
                var tree = trees[i, p];
                if (!tree.RootExpanded || tree.RootVisits < sims[p]) continue;
                int move = tree.ChooseMove(0f);
                trees[i, 0].Advance(move);
                trees[i, 1].Advance(move);
                var next = trees[i, 0].RootState;
                if (!next.Outcome.IsOver()) continue;
                // The player who just moved (p) won, unless it is a draw.
                if (next.Outcome == Outcome.Draw) draws++;
                else if (p == 0) winsA++;
                else winsB++;
                done++;
                progress?.Invoke(new MatchResult(winsA, draws, winsB));
                if (nextGame < cfg.Games)
                {
                    aFirst[i] = nextGame % 2 == 0;
                    var start = Opening(nextGame);
                    nextGame++;
                    trees[i, 0].Reset(start);
                    trees[i, 1].Reset(start);
                }
                else active[i] = false;
            }
        }
        return new MatchResult(winsA, draws, winsB);
    }

    /// <summary>
    /// A match between arbitrary players (e.g. random, perfect solver, rollout MCTS), games run in
    /// parallel on the CPU with one player instance per game from the factories.
    /// </summary>
    public static MatchResult PlayPlayers<TGame>(Func<int, IPlayer<TGame>> makeA, Func<int, IPlayer<TGame>> makeB, int games,
                                                 int randomOpeningMoves = 0, int seed = 5)
        where TGame : struct, IGame<TGame>
    {
        int winsA = 0, draws = 0, winsB = 0;
        System.Threading.Tasks.Parallel.For(0, games, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, g =>
        {
            var pa = makeA(g);
            var pb = makeB(g);
            pa.NewGame();
            pb.NewGame();
            var rng = new Random(seed * 1000 + g / 2);   // games 2k and 2k+1 share their random opening, colours swapped
            bool aFirst = g % 2 == 0;
            var state = TGame.Initial;
            Span<int> moves = stackalloc int[TGame.ActionCount];
            int lastMover = 0;
            while (!state.Outcome.IsOver())
            {
                bool aToMove = (state.SideToMove == 0) == aFirst;
                int move;
                if (state.Ply < randomOpeningMoves)
                {
                    int n = state.LegalMoves(moves);
                    move = moves[rng.Next(n)];
                }
                else move = aToMove ? pa.ChooseMove(state) : pb.ChooseMove(state);
                lastMover = aToMove ? 0 : 1;
                state.Play(move);
            }
            if (state.Outcome == Outcome.Draw) Interlocked.Increment(ref draws);
            else if (lastMover == 0) Interlocked.Increment(ref winsA);
            else Interlocked.Increment(ref winsB);
        });
        return new MatchResult(winsA, draws, winsB);
    }
}
