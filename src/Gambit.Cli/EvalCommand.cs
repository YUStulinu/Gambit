using Gambit.Core.Games;
using Gambit.Core.Neural;
using Gambit.Core.Rating;
using Gambit.Core.Search;
using Gambit.Core.Solver;
using Gambit.Neural;
using TorchSharp;
using static TorchSharp.torch;

namespace Gambit.Cli;

/// <summary>Loads a network from a .pt checkpoint (GPU) or a .gnet file (pure C#).</summary>
public static class Models
{
    public static IEvaluator<TGame> Load<TGame>(string path, bool half = true) where TGame : unmanaged, IGame<TGame>
    {
        if (path.EndsWith(".gnet", StringComparison.OrdinalIgnoreCase)) return NativeNetwork.Load(path).For<TGame>();
        var device = cuda.is_available() ? CUDA : CPU;
        var net = Checkpoints.Load(path, device);
        return new TorchEvaluator<TGame>(net, device, half);
    }
}

/// <summary>
/// gambit eval --game connect4 --model runs/connect4/models/best.pt [--sims 200] [--baselines]
/// Grades a network against the perfect solver (Connect Four) and plays it against the
/// rollout-MCTS anchors. --baselines also grades simple reference players on the same positions.
/// </summary>
public sealed class EvalCommand : IGameCommand
{
    public int Run<TGame>(Args args) where TGame : unmanaged, IGame<TGame>
    {
        string? model = args.GetOrNull("model");
        int sims = args.Int("sims", 200);
        int games = args.Int("games", 100);
        bool baselines = args.Flag("baselines");
        string setPath = args.Get("set", Path.Combine("data", "connect4-solved.txt"));
        string? run = args.GetOrNull("run");
        int every = args.Int("every", 5);
        int perfectGames = args.Int("perfect-games", 60);
        args.CheckUnused();
        if (run != null)
        {
            if (typeof(TGame) != typeof(ConnectFour)) throw new ArgumentException("--run evaluation needs the Connect Four solver");
            return EvaluateRun(run, every, sims, perfectGames, SolvedPositions.Load(setPath));
        }

        if (typeof(TGame) == typeof(ConnectFour))
        {
            List<SolvedPosition> set;
            if (File.Exists(setPath)) set = SolvedPositions.Load(setPath);
            else
            {
                Console.WriteLine($"building {setPath} (1000 solved positions)...");
                set = SolvedPositions.Generate(1000, 10, 30, 2024);
                Directory.CreateDirectory(Path.GetDirectoryName(setPath)!);
                SolvedPositions.Save(setPath, set);
            }
            Console.WriteLine($"{set.Count} solved positions, plies {set.Min(p => p.Moves.Length)}-{set.Max(p => p.Moves.Length)}, " +
                              $"{100 * set.Average(p => p.Scores.Count(s => s == p.Best) / (double)p.Scores.Count(s => s.HasValue)):F1}% of legal moves are optimal on average");
            if (baselines)
            {
                double randomOptimal = set.Average(p => p.Scores.Count(s => s == p.Best) / (double)p.Scores.Count(s => s.HasValue));
                double randomKeeps = set.Average(p => p.Scores.Count(s => s.HasValue && Math.Sign(s.Value) == p.Result) / (double)p.Scores.Count(s => s.HasValue));
                Console.WriteLine($"random move: optimal {100 * randomOptimal:F1}%, result-preserving {100 * randomKeeps:F1}%");
                foreach (int n in new[] { 200, 2000, 20000 })
                {
                    var acc = SolvedPositions.Grade(set, new ParallelRolloutEvaluator<ConnectFour>(1), n);
                    Console.WriteLine($"pure MCTS, {n} rollouts: {acc}");
                }
            }
            if (model != null)
            {
                var ev = Models.Load<ConnectFour>(model);
                Console.WriteLine($"network, raw policy: {SolvedPositions.Grade(set, ev, 0)}");
                Console.WriteLine($"network + MCTS {sims}: {SolvedPositions.Grade(set, ev, sims)}");
                Console.WriteLine($"network + MCTS {sims * 4}: {SolvedPositions.Grade(set, ev, sims * 4)}");
            }
        }

        if (model != null)
        {
            var ev = Models.Load<TGame>(model);
            foreach (var (anchor, n) in TrainCommand.Anchors.Append(("mcts-rollout-20000", 20000)))
            {
                var r = Arena.Play<TGame>(ev, new ParallelRolloutEvaluator<TGame>(3), new ArenaConfig
                {
                    Games = games, SimulationsA = sims, SimulationsB = n, Parallel = games, LeavesPerStepA = 4, LeavesPerStepB = 16,
                });
                Console.WriteLine($"network ({sims} sims) vs {anchor}: {r}");
            }
        }
        return 0;
    }

    /// <summary>Grades every <paramref name="every"/>-th saved generation of a run; writes evaluation.tsv.</summary>
    static int EvaluateRun(string run, int every, int sims, int perfectGames, List<SolvedPosition> set)
    {
        var models = Directory.GetFiles(Path.Combine(run, "models"), "gen-*.pt")
            .Select(p => (Path: p, Gen: int.Parse(Path.GetFileNameWithoutExtension(p)[4..]))).OrderBy(m => m.Gen).ToList();
        var chosen = models.Where((m, i) => m.Gen % every == 0 || i == models.Count - 1 || m.Gen == 0)
                           .GroupBy(m => m.Gen / every).Select(g => g.Last()).ToList();
        if (chosen[^1].Gen != models[^1].Gen) chosen.Add(models[^1]);
        string path = Path.Combine(run, "evaluation.tsv");
        var lines = new List<string> { "generation	blunders_raw	blunders_mcts	keeps_mcts	optimal_mcts	value_sign	score_vs_perfect" };
        foreach (var (file, gen) in chosen)
        {
            var ev = Models.Load<ConnectFour>(file);
            var raw = SolvedPositions.Grade(set, ev, 0);
            var mcts = SolvedPositions.Grade(set, ev, sims);
            var locked = new LockedEvaluator<ConnectFour>(ev);
            var vsPerfect = Arena.PlayPlayers<ConnectFour>(_ => new MctsPlayer<ConnectFour>(locked, sims, 16), _ => new SolverPlayer(),
                                                           perfectGames, 10);
            Console.WriteLine($"gen-{gen:D4}: raw {raw}");
            Console.WriteLine($"          MCTS {sims}: {mcts}; vs perfect play: {vsPerfect}");
            lines.Add(string.Join('	', new object[] { gen, raw.BlunderRate, mcts.BlunderRate, mcts.KeepsResult, mcts.Optimal,
                                                       mcts.ValueSign, vsPerfect.ScoreA }
                .Select(o => Convert.ToString(o, System.Globalization.CultureInfo.InvariantCulture))));
            File.WriteAllLines(path, lines);
        }
        Console.WriteLine($"-> {path}");
        return 0;
    }
}
