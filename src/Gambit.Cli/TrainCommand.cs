using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Gambit.Core.Games;
using Gambit.Core.Neural;
using Gambit.Core.Rating;
using Gambit.Core.Search;
using Gambit.Core.Solver;
using Gambit.Core.Training;
using Gambit.Neural;
using TorchSharp;
using static TorchSharp.torch;

namespace Gambit.Cli;

/// <summary>
/// gambit train --game connect4 --run runs/connect4 [--generations 100] ...
///
/// The AlphaZero loop, one generation at a time:
///   1. self-play: the current best network plays thousands of games against itself;
///   2. training: the network being trained learns from a window of recent self-play positions;
///   3. gating: a snapshot of it (the candidate) plays a match against the best network and
///      replaces it only if it scores at least --gate (55%);
///   4. rating: arena results feed an Elo table anchored on fixed opponents (pure MCTS with random
///      rollouts), and for Connect Four the best network is graded against the perfect solver.
/// Everything is saved in the run directory, and an interrupted run resumes where it stopped.
/// </summary>
public sealed class TrainCommand : IGameCommand
{
    sealed record Options
    {
        public string Run = "";
        public int Generations, GamesPerGeneration, Parallel, Simulations, FastSimulations, TemperatureMoves, RandomOpening;
        public float FullSearchProbability, DirichletAlpha, ValueMix, Gate;
        public int Blocks, Channels, BatchSize, ReplayWindow, ArenaGames, ArenaSimulations, EvalEvery, AnchorGames;
        public double LearningRate, SampleReuse;
        public bool Half;
        public int Seed;
    }

    static Options Parse<TGame>(Args a) where TGame : unmanaged, IGame<TGame>
    {
        bool c4 = typeof(TGame) == typeof(ConnectFour);
        return new Options
        {
            Run = a.Get("run", Path.Combine("runs", TGame.Name)),
            Generations = a.Int("generations", 60),
            GamesPerGeneration = a.Int("games", c4 ? 4096 : 2048),
            Parallel = a.Int("parallel", 2048),
            Simulations = a.Int("sims", c4 ? 200 : 300),
            FastSimulations = a.Int("fast-sims", c4 ? 40 : 60),
            FullSearchProbability = a.Float("full-prob", 0.25f),
            TemperatureMoves = a.Int("temperature-moves", c4 ? 8 : 12),
            RandomOpening = a.Int("random-opening", c4 ? 4 : 2),
            DirichletAlpha = a.Float("alpha", c4 ? 1.0f : 0.2f),
            ValueMix = a.Float("value-mix", 0.0f),
            Gate = a.Float("gate", 0.95f),
            Blocks = a.Int("blocks", 6),
            Channels = a.Int("channels", 64),
            BatchSize = a.Int("batch", 512),
            LearningRate = a.Double("lr", 0.02),
            ReplayWindow = a.Int("window", c4 ? 250_000 : 300_000),
            SampleReuse = a.Double("reuse", 4.0),
            ArenaGames = a.Int("arena-games", 200),
            ArenaSimulations = a.Int("arena-sims", 200),
            EvalEvery = a.Int("eval-every", 5),
            AnchorGames = a.Int("anchor-games", 100),
            Half = !a.Flag("fp32"),
            Seed = a.Int("seed", 1),
        };
    }

    /// <summary>One line of the run log (log.jsonl).</summary>
    public sealed record GenerationLog
    {
        public int Generation { get; init; }
        public int BestGeneration { get; init; }
        public bool Accepted { get; init; }
        public double ArenaScore { get; init; }
        public string Arena { get; init; } = "";
        public int Games { get; init; }
        public long Samples { get; init; }
        public int ReplaySize { get; init; }
        public double FirstWinRate { get; init; }
        public double DrawRate { get; init; }
        public double AverageLength { get; init; }
        public double GamesPerSecond { get; init; }
        public double SimulationsPerSecond { get; init; }
        public double PolicyLoss { get; init; }
        public double ValueLoss { get; init; }
        public double TargetEntropy { get; init; }
        public double Elo { get; init; }
        public double? SolverOptimal { get; init; }
        public double? SolverKeepsResult { get; init; }
        public double? SolverOptimalRaw { get; init; }
        public double SelfPlaySeconds { get; init; }
        public double TrainSeconds { get; init; }
        public double ArenaSeconds { get; init; }
        public double ElapsedSeconds { get; init; }
    }

    public int Run<TGame>(Args args) where TGame : unmanaged, IGame<TGame>
    {
        var o = Parse<TGame>(args);
        args.CheckUnused();
        if (!cuda.is_available()) Console.WriteLine("warning: CUDA not available, training on the CPU will be slow");
        var device = cuda.is_available() ? CUDA : CPU;
        torch.random.manual_seed(o.Seed);
        Directory.CreateDirectory(Path.Combine(o.Run, "models"));
        Directory.CreateDirectory(Path.Combine(o.Run, "games"));
        string logPath = Path.Combine(o.Run, "log.jsonl"), eloPath = Path.Combine(o.Run, "elo.tsv");
        string bestPath = Path.Combine(o.Run, "models", "best.pt"), trainPath = Path.Combine(o.Run, "models", "latest.pt");
        File.WriteAllText(Path.Combine(o.Run, "options.json"),
                          JsonSerializer.Serialize(o, new JsonSerializerOptions { WriteIndented = true, IncludeFields = true }));

        // ---- state: resume or start ---------------------------------------------------------------------
        var logs = File.Exists(logPath)
            ? File.ReadLines(logPath).Select(l => JsonSerializer.Deserialize<GenerationLog>(l)!).ToList()
            : [];
        AlphaZeroNet trainNet, best;
        int bestGen;
        var replay = new ReplayBuffer<TGame>(o.ReplayWindow);
        if (logs.Count > 0 && File.Exists(trainPath) && File.Exists(bestPath))
        {
            trainNet = Checkpoints.Load(trainPath, device);
            best = Checkpoints.Load(bestPath, device);
            bestGen = logs[^1].BestGeneration;
            foreach (var f in Directory.GetFiles(Path.Combine(o.Run, "games"), "gen-*.bin").Order().TakeLast(40))
                foreach (var g in ReplayBuffer<TGame>.Load(f)) replay.AddGame(g);
            Console.WriteLine($"resuming after generation {logs[^1].Generation} (best: generation {bestGen}, replay {replay.Count:N0} positions)");
        }
        else
        {
            logs.Clear();
            File.Delete(logPath);
            File.Delete(eloPath);
            trainNet = new AlphaZeroNet(NetworkSpec.For<TGame>(o.Blocks, o.Channels)).to(device);
            best = Checkpoints.Clone(trainNet, device);
            bestGen = 0;
            Checkpoints.Save(best, Path.Combine(o.Run, "models", "gen-0000.pt"));
            Checkpoints.Save(best, bestPath);
            Checkpoints.Save(trainNet, trainPath);
            Checkpoints.Export(best, Path.Combine(o.Run, "models", "gen-0000.gnet"));
        }
        var elo = EloTable.Load(eloPath);
        var trainer = new Trainer<TGame>(trainNet, device,
                                         new TrainConfig { BatchSize = o.BatchSize, LearningRate = o.LearningRate }, o.Seed);
        List<SolvedPosition>? solved = typeof(TGame) == typeof(ConnectFour) ? LoadOrBuildSolverSet(o.Run) : null;

        Console.WriteLine($"{TGame.Name}: {trainNet.Spec} ({trainNet.ParameterCount:N0} parameters) on {device.type}, " +
                          $"{o.GamesPerGeneration} games/generation, {o.Simulations}/{o.FastSimulations} simulations");
        double elapsedBefore = logs.Count > 0 ? logs[^1].ElapsedSeconds : 0;
        var clock = Stopwatch.StartNew();
        int startGen = logs.Count > 0 ? logs[^1].Generation + 1 : 1;
        if (logs.Count == 0) EvaluateAnchors<TGame>(best, "gen-0000", device, o, elo);

        for (int gen = startGen; gen <= o.Generations; gen++)
        {
            Console.WriteLine($"\n=== generation {gen} (best: {bestGen}) ===");

            // 1. Self-play with the best network.
            var sw = Stopwatch.StartNew();
            var stats = new SelfPlayStats();
            List<GameRecord<TGame>> games;
            using (var ev = new TorchEvaluator<TGame>(best, device, o.Half))
            {
                var cached = new CachedEvaluator<TGame>(ev, 20);
                var engine = new SelfPlayEngine<TGame>(cached, new SelfPlayConfig
                {
                    ParallelGames = o.Parallel, Simulations = o.Simulations, FastSimulations = o.FastSimulations,
                    FullSearchProbability = o.FullSearchProbability, TemperatureMoves = o.TemperatureMoves,
                    DirichletAlpha = o.DirichletAlpha, ValueTargetMix = o.ValueMix, Seed = o.Seed * 100_003 + gen,
                    RandomOpeningMoves = o.RandomOpening,
                });
                games = engine.Play(o.GamesPerGeneration, stats);
            }
            ReplayBuffer<TGame>.Save(Path.Combine(o.Run, "games", $"gen-{gen:D4}.bin"), games);
            foreach (var g in games) replay.AddGame(g);
            double selfPlaySeconds = sw.Elapsed.TotalSeconds;
            Console.WriteLine($"self-play: {stats} ({selfPlaySeconds:F0}s)");

            // 2. Train (step learning-rate schedule: x0.3 after 60% of the generations, x0.1 after 85%).
            sw.Restart();
            trainer.LearningRate = o.LearningRate * (gen > 0.85 * o.Generations ? 0.1 : gen > 0.6 * o.Generations ? 0.3 : 1.0);
            int steps = (int)Math.Ceiling(stats.Samples * o.SampleReuse / o.BatchSize);
            var metrics = trainer.Train(replay, steps);
            Checkpoints.Save(trainNet, trainPath);
            double trainSeconds = sw.Elapsed.TotalSeconds;
            Console.WriteLine($"training: {metrics}, lr {trainer.LearningRate:G3} (replay {replay.Count:N0} positions)");

            // 3. Gate: candidate against the best network.
            sw.Restart();
            MatchResult arena;
            using (var cand = new TorchEvaluator<TGame>(trainNet, device, o.Half))
            using (var champ = new TorchEvaluator<TGame>(best, device, o.Half))
            {
                arena = Arena.Play<TGame>(cand, champ, new ArenaConfig
                {
                    Games = o.ArenaGames, SimulationsA = o.ArenaSimulations, SimulationsB = o.ArenaSimulations,
                    Parallel = o.ArenaGames, Seed = gen, LeavesPerStepA = 4, LeavesPerStepB = 4,
                    RandomOpeningMoves = o.RandomOpening,
                });
            }
            double arenaSeconds = sw.Elapsed.TotalSeconds;
            string candName = $"gen-{gen:D4}", bestName = $"gen-{bestGen:D4}";
            elo.Add(candName, bestName, arena.WinsA, arena.Draws, arena.WinsB);
            // Gate on the likelihood of superiority (draws are common between strong players).
            bool accepted = arena.WinsA > arena.WinsB && arena.LikelihoodOfSuperiority >= o.Gate;
            Console.WriteLine($"arena {candName} vs {bestName}: {arena} -> {(accepted ? "ACCEPTED" : "rejected")}");
            if (accepted)
            {
                best.Dispose();
                best = Checkpoints.Clone(trainNet, device);
                bestGen = gen;
                Checkpoints.Save(best, bestPath);
                Checkpoints.Save(best, Path.Combine(o.Run, "models", $"{candName}.pt"));
                Checkpoints.Export(best, Path.Combine(o.Run, "models", $"{candName}.gnet"));
                Checkpoints.Export(best, Path.Combine(o.Run, $"{TGame.Name}-best.gnet"));
            }

            // 4. Anchors and solver accuracy, every few generations.
            SolverAccuracy? acc = null, accRaw = null;
            if (gen % o.EvalEvery == 0 || gen == o.Generations)
            {
                EvaluateAnchors<TGame>(best, $"gen-{bestGen:D4}", device, o, elo);
                if (solved != null)
                {
                    using var ev = new TorchEvaluator<ConnectFour>(best, device, o.Half);
                    acc = SolvedPositions.Grade(solved, ev, o.ArenaSimulations);
                    accRaw = SolvedPositions.Grade(solved, ev, 0);
                    Console.WriteLine($"vs perfect play, MCTS {o.ArenaSimulations}: {acc}");
                    Console.WriteLine($"vs perfect play, raw network: {accRaw}");
                }
            }
            elo.Save(eloPath);
            var ratings = elo.Compute(Anchors[0].Name, 0);
            double rating = ratings.GetValueOrDefault($"gen-{bestGen:D4}", double.NaN);
            Console.WriteLine($"Elo of the best network (generation {bestGen}): {rating:F0} (pure MCTS with {Anchors[0].Simulations} simulations = 0)");

            var line = new GenerationLog
            {
                Generation = gen, BestGeneration = bestGen, Accepted = accepted, ArenaScore = arena.ScoreA,
                Arena = arena.ToString(), Games = stats.Games, Samples = stats.Samples, ReplaySize = replay.Count,
                FirstWinRate = stats.FirstWins / (double)stats.Games, DrawRate = stats.Draws / (double)stats.Games,
                AverageLength = stats.AverageLength, GamesPerSecond = stats.GamesPerSecond,
                SimulationsPerSecond = stats.SimulationsPerSecond, PolicyLoss = metrics.PolicyLoss, ValueLoss = metrics.ValueLoss,
                TargetEntropy = metrics.PolicyEntropy, Elo = rating, SolverOptimal = acc?.Optimal,
                SolverKeepsResult = acc?.KeepsResult, SolverOptimalRaw = accRaw?.Optimal, SelfPlaySeconds = selfPlaySeconds,
                TrainSeconds = trainSeconds, ArenaSeconds = arenaSeconds, ElapsedSeconds = elapsedBefore + clock.Elapsed.TotalSeconds,
            };
            File.AppendAllText(logPath, JsonSerializer.Serialize(line) + "\n");
            logs.Add(line);
            Charts.WriteProgress(Path.Combine(o.Run, "progress.svg"), TGame.Name, logs, elo.Compute(Anchors[0].Name, 0));
        }
        Console.WriteLine($"\ndone: best network is generation {bestGen} -> {Path.Combine(o.Run, TGame.Name + "-best.gnet")}");
        return 0;
    }

    /// <summary>Fixed opponents that anchor the Elo scale: MCTS with random-rollout evaluation.</summary>
    public static readonly (string Name, int Simulations)[] Anchors = [("mcts-rollout-200", 200), ("mcts-rollout-2000", 2000)];

    static void EvaluateAnchors<TGame>(AlphaZeroNet net, string name, Device device, Options o, EloTable elo)
        where TGame : unmanaged, IGame<TGame>
    {
        foreach (var (anchor, sims) in Anchors)
        {
            var sw = Stopwatch.StartNew();
            using var ev = new TorchEvaluator<TGame>(net, device, o.Half);
            // The rollout evaluator runs on the CPU; one instance per thread would be needed to share it,
            // so the batched arena calls it from one thread with batches of positions.
            var rollout = new ParallelRolloutEvaluator<TGame>(o.Seed);
            var r = Arena.Play<TGame>(ev, rollout, new ArenaConfig
            {
                Games = o.AnchorGames, SimulationsA = o.ArenaSimulations, SimulationsB = sims, Parallel = o.AnchorGames, Seed = 7,
                LeavesPerStepA = 4, LeavesPerStepB = 16,
            });
            elo.Add(name, anchor, r.WinsA, r.Draws, r.WinsB);
            Console.WriteLine($"anchor {name} vs {anchor}: {r} ({sw.Elapsed.TotalSeconds:F0}s)");
        }
    }

    static List<SolvedPosition> LoadOrBuildSolverSet(string run)
    {
        string path = Path.Combine("data", "connect4-solved.txt");
        if (!File.Exists(path)) path = Path.Combine(run, "connect4-solved.txt");
        if (File.Exists(path)) return SolvedPositions.Load(path);
        Console.WriteLine("building the solved test set (1000 positions, a few minutes, once)...");
        var set = SolvedPositions.Generate(1000, 10, 30, 2024);
        SolvedPositions.Save(path, set);
        return set;
    }
}

/// <summary>Random-rollout evaluation spread over all cores (for the anchor opponents).</summary>
public sealed class ParallelRolloutEvaluator<TGame>(int seed) : IEvaluator<TGame> where TGame : struct, IGame<TGame>
{
    readonly ThreadLocal<RolloutEvaluator<TGame>> _local = new(() => new RolloutEvaluator<TGame>(1, seed + Environment.CurrentManagedThreadId));

    public void Evaluate(ReadOnlySpan<TGame> states, Span<float> logits, Span<float> values)
    {
        var st = states.ToArray();
        var v = new float[st.Length];
        Parallel.For(0, st.Length, i =>
        {
            Span<float> l = stackalloc float[TGame.ActionCount];
            Span<float> one = stackalloc float[1];
            _local.Value!.Evaluate(st.AsSpan(i, 1), l, one);
            v[i] = one[0];
        });
        logits[..(st.Length * TGame.ActionCount)].Clear();
        v.CopyTo(values);
    }
}
