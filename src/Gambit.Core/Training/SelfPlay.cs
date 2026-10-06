using System.Diagnostics;
using Gambit.Core.Games;
using Gambit.Core.Search;

namespace Gambit.Core.Training;

/// <summary>One position from self-play with its training targets.</summary>
public sealed class Sample<TGame>(TGame state, float[] policy, float rootValue) where TGame : struct, IGame<TGame>
{
    public TGame State = state;
    /// <summary>Visit distribution of the search at this position: the policy target.</summary>
    public readonly float[] Policy = policy;
    /// <summary>The search's value of the position (side to move), in [-1, 1].</summary>
    public readonly float RootValue = rootValue;
    /// <summary>The value training target: the game result for the side to move here (-1, 0 or 1),
    /// optionally mixed with <see cref="RootValue"/>.</summary>
    public float ValueTarget;
}

/// <summary>A finished self-play game.</summary>
public sealed class GameRecord<TGame> where TGame : struct, IGame<TGame>
{
    public readonly List<int> Moves = [];
    public readonly List<Sample<TGame>> Samples = [];
    /// <summary>1 if the first player won, -1 if the second did, 0 for a draw.</summary>
    public int Result;
}

public sealed record SelfPlayConfig
{
    /// <summary>Games played at the same time; their leaves are evaluated together in one batch.</summary>
    public int ParallelGames { get; init; } = 512;

    /// <summary>Simulations for a full search (the positions that become training samples).</summary>
    public int Simulations { get; init; } = 400;

    /// <summary>Playout cap randomisation (KataGo): with probability 1 - FullSearchProbability a move
    /// gets only FastSimulations, no noise, and produces no sample - games finish much faster while
    /// the recorded positions keep full-strength targets.</summary>
    public int FastSimulations { get; init; } = 80;
    public float FullSearchProbability { get; init; } = 0.25f;

    /// <summary>Leaves selected per game per batch (virtual loss spreads them out).</summary>
    public int LeavesPerGame { get; init; } = 1;

    /// <summary>Moves at the start of a game played proportionally to visits (temperature 1); later moves are greedy.</summary>
    public int TemperatureMoves { get; init; } = 10;

    /// <summary>Every game starts with a uniformly random number (0 to this) of uniformly random moves.
    /// Without it, once the policy becomes sharp, sampling by visits keeps choosing the same opening
    /// and self-play collapses into thousands of copies of one game; with it the network has to learn
    /// to win the won positions and hold the drawn ones of every kind of opening.</summary>
    public int RandomOpeningMoves { get; init; } = 0;

    /// <summary>Value target = (1 - mix) * game result + mix * search value.</summary>
    public float ValueTargetMix { get; init; } = 0.0f;

    public float DirichletAlpha { get; init; } = 1.0f;
    public int Seed { get; init; } = 1;
}

public sealed class SelfPlayStats
{
    public int Games, FirstWins, SecondWins, Draws;
    public long Moves, Samples, Simulations, Evaluations, Batches;
    public double Seconds, EvalSeconds, SelectSeconds, ExpandSeconds;
    public double GamesPerSecond => Games / Math.Max(Seconds, 1e-9);
    public double SimulationsPerSecond => Simulations / Math.Max(Seconds, 1e-9);
    public double AverageBatch => Evaluations / (double)Math.Max(Batches, 1);
    public double AverageLength => Moves / (double)Math.Max(Games, 1);

    public override string ToString() =>
        $"{Games} games ({FirstWins}/{Draws}/{SecondWins} first/draw/second), {AverageLength:F1} moves avg, " +
        $"{Samples} samples, {GamesPerSecond:F1} games/s, {SimulationsPerSecond / 1000:F1}k sims/s, " +
        $"batch {AverageBatch:F0}; time: select {100 * SelectSeconds / Math.Max(Seconds, 1e-9):F0}%, " +
        $"evaluate {100 * EvalSeconds / Math.Max(Seconds, 1e-9):F0}%, expand {100 * ExpandSeconds / Math.Max(Seconds, 1e-9):F0}%";
}

/// <summary>
/// Plays many self-play games at once. Every step:
///   1. (parallel over games) each game's tree selects a leaf to evaluate;
///   2. all those leaves go to the evaluator in one batch (one GPU call);
///   3. (parallel over games) each tree expands its leaf; a game whose search is complete plays its
///      move, and a finished game is handed to the caller and replaced by a new one.
/// </summary>
public sealed class SelfPlayEngine<TGame> where TGame : struct, IGame<TGame>
{
    sealed class Slot
    {
        public required SearchTree<TGame> Tree;
        public required Random Rng;
        public GameRecord<TGame> Record = new();
        public int Target;          // simulations wanted for the current move
        public bool Full;           // full search (sample) or fast move
        public bool NoiseDone;
        public int LeafCount;
        public readonly int[] Leaves = new int[8];
        public readonly TGame[] States = new TGame[8];
        public bool Active = true;
    }

    readonly IEvaluator<TGame> _eval;
    readonly SelfPlayConfig _cfg;

    public SelfPlayEngine(IEvaluator<TGame> evaluator, SelfPlayConfig config)
    {
        _eval = evaluator;
        _cfg = config;
        if (config.LeavesPerGame is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(config), "LeavesPerGame must be 1..8");
    }

    /// <summary>Plays <paramref name="games"/> games and returns them (with statistics in <paramref name="stats"/>).</summary>
    public List<GameRecord<TGame>> Play(int games, SelfPlayStats stats, Action<GameRecord<TGame>>? onGame = null,
                                        CancellationToken cancel = default)
    {
        var finished = new List<GameRecord<TGame>>(games);
        var searchCfg = SearchConfig.SelfPlay(_cfg.DirichletAlpha);
        int parallel = Math.Min(_cfg.ParallelGames, games);
        var slots = new Slot[parallel];
        int started = 0;
        for (int i = 0; i < parallel; i++)
        {
            slots[i] = new Slot
            {
                Tree = new SearchTree<TGame>(searchCfg, _cfg.Seed * 7919 + i),
                Rng = new Random(_cfg.Seed * 104729 + i),
            };
            NewGame(slots[i]);
            StartMove(slots[i]);
            started++;
        }

        int a = TGame.ActionCount;
        int maxBatch = parallel * _cfg.LeavesPerGame;
        var batchStates = new TGame[maxBatch];
        var batchLogits = new float[maxBatch * a];
        var batchValues = new float[maxBatch];
        var offsets = new int[parallel];
        var sw = Stopwatch.StartNew();
        var evalWatch = new Stopwatch();
        var selectWatch = new Stopwatch();
        var expandWatch = new Stopwatch();
        object gate = new();

        while (finished.Count < games && !cancel.IsCancellationRequested)
        {
            // 1. Select leaves.
            selectWatch.Start();
            Parallel.For(0, parallel, i =>
            {
                var s = slots[i];
                s.LeafCount = 0;
                if (!s.Active) return;
                int want = s.Tree.RootExpanded ? Math.Min(_cfg.LeavesPerGame, s.Target - s.Tree.RootVisits) : 1;
                for (int k = 0; k < Math.Max(want, 1); k++)
                {
                    var kind = s.Tree.SelectLeaf(out int leaf, out var st);
                    if (kind == LeafKind.Collision) break;
                    if (kind == LeafKind.NeedsEvaluation)
                    {
                        s.Leaves[s.LeafCount] = leaf;
                        s.States[s.LeafCount++] = st;
                    }
                    else if (s.Tree.RootVisits >= s.Target && s.Tree.RootExpanded) break;
                }
            });
            int total = 0;
            for (int i = 0; i < parallel; i++)
            {
                offsets[i] = total;
                for (int k = 0; k < slots[i].LeafCount; k++) batchStates[total + k] = slots[i].States[k];
                total += slots[i].LeafCount;
            }
            selectWatch.Stop();

            // 2. Evaluate the whole batch at once.
            if (total > 0)
            {
                evalWatch.Start();
                _eval.Evaluate(batchStates.AsSpan(0, total), batchLogits, batchValues);
                evalWatch.Stop();
                stats.Evaluations += total;
                stats.Batches++;
            }

            // 3. Expand, play moves, finish games.
            expandWatch.Start();
            Parallel.For(0, parallel, i =>
            {
                var s = slots[i];
                if (!s.Active) return;
                for (int k = 0; k < s.LeafCount; k++)
                {
                    int j = offsets[i] + k;
                    s.Tree.Expand(s.Leaves[k], s.States[k], batchLogits.AsSpan(j * a, a), batchValues[j]);
                }
                if (!s.Tree.RootExpanded) return;
                if (s.Full && !s.NoiseDone)
                {
                    s.Tree.AddRootNoise();
                    s.NoiseDone = true;
                }
                if (s.Tree.RootVisits < s.Target) return;

                Interlocked.Add(ref stats.Simulations, s.Target);
                var root = s.Tree.RootState;
                if (s.Full)
                {
                    var policy = new float[a];
                    s.Tree.VisitDistribution(policy);
                    s.Record.Samples.Add(new Sample<TGame>(root, policy, s.Tree.RootValue));
                }
                float temperature = root.Ply < _cfg.TemperatureMoves ? 1f : 0f;
                int move = s.Tree.ChooseMove(temperature);
                s.Record.Moves.Add(move);
                s.Tree.Advance(move);
                var next = s.Tree.RootState;
                if (next.Outcome.IsOver())
                {
                    FinishGame(s.Record, next);
                    lock (gate)
                    {
                        if (finished.Count < games)
                        {
                            finished.Add(s.Record);
                            onGame?.Invoke(s.Record);
                        }
                        if (started < games) started++;
                        else s.Active = false;
                    }
                    s.Record = new GameRecord<TGame>();
                    NewGame(s);
                }
                StartMove(s);
            });
            expandWatch.Stop();
        }

        stats.Seconds += sw.Elapsed.TotalSeconds;
        stats.EvalSeconds += evalWatch.Elapsed.TotalSeconds;
        stats.SelectSeconds += selectWatch.Elapsed.TotalSeconds;
        stats.ExpandSeconds += expandWatch.Elapsed.TotalSeconds;
        foreach (var g in finished)
        {
            stats.Games++;
            stats.Moves += g.Moves.Count;
            stats.Samples += g.Samples.Count;
            if (g.Result > 0) stats.FirstWins++;
            else if (g.Result < 0) stats.SecondWins++;
            else stats.Draws++;
        }
        return finished;
    }

    /// <summary>Resets the slot to a new game, after its random opening moves (recorded, not sampled).</summary>
    void NewGame(Slot s)
    {
        var state = TGame.Initial;
        int random = _cfg.RandomOpeningMoves > 0 ? s.Rng.Next(_cfg.RandomOpeningMoves + 1) : 0;
        Span<int> moves = stackalloc int[TGame.ActionCount];
        for (int k = 0; k < random && !state.Outcome.IsOver(); k++)
        {
            int n = state.LegalMoves(moves);
            int m = moves[s.Rng.Next(n)];
            state.Play(m);
            s.Record.Moves.Add(m);
        }
        if (state.Outcome.IsOver())
        {
            state = TGame.Initial;
            s.Record.Moves.Clear();
        }
        s.Tree.Reset(state);
    }

    void StartMove(Slot s)
    {
        s.Full = s.Rng.NextDouble() < _cfg.FullSearchProbability;
        s.Target = s.Full ? _cfg.Simulations : _cfg.FastSimulations;
        s.NoiseDone = false;
    }

    void FinishGame(GameRecord<TGame> record, in TGame final)
    {
        // Loss at the end means the player who made the last move won.
        int lastMover = (final.Ply - 1) & 1;
        record.Result = final.Outcome == Outcome.Draw ? 0 : lastMover == 0 ? 1 : -1;
        foreach (var smp in record.Samples)
        {
            float z = record.Result == 0 ? 0f : (smp.State.SideToMove == 0 ? record.Result : -record.Result);
            smp.ValueTarget = (1 - _cfg.ValueTargetMix) * z + _cfg.ValueTargetMix * smp.RootValue;
        }
    }
}
