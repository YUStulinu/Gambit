using Gambit.Core.Games;
using Gambit.Core.Rating;
using Gambit.Core.Search;
using Gambit.Neural;
using static TorchSharp.torch;

namespace Gambit.Cli;

/// <summary>
/// gambit arena --game connect4 --a runs/connect4/models/gen-0040.pt --b rollout:2000 [--games 200 --sims 200]
/// A match between two players. A player is a checkpoint (.pt), an exported network (.gnet),
/// "rollout:N" (pure MCTS with random rollouts and N simulations), or "perfect" (the Connect Four solver,
/// which plays from move --perfect-from on, after random opening moves).
/// </summary>
public sealed class ArenaCommand : IGameCommand
{
    public int Run<TGame>(Args args) where TGame : unmanaged, IGame<TGame>
    {
        string a = args.Get("a", ""), b = args.Get("b", "rollout:2000");
        int games = args.Int("games", 200), sims = args.Int("sims", 200), simsB = args.Int("sims-b", sims);
        int openings = args.Int("random-opening", 2);
        args.CheckUnused();
        if (a == "") throw new ArgumentException("--a is required");
        if (b == "perfect")
        {
            if (typeof(TGame) != typeof(ConnectFour)) throw new ArgumentException("the perfect player exists for connect4 only");
            var ev = Models.Load<ConnectFour>(a);
            // The solver needs seconds for positions with fewer than ~10 stones, so both sides play
            // `openings` random moves first; each opening is played twice with colours swapped.
            var r = Arena.PlayPlayers<ConnectFour>(_ => new MctsPlayer<ConnectFour>(ev, sims, 16, name: "network"),
                                                   _ => new SolverPlayer(), games, Math.Max(openings, 8));
            Console.WriteLine($"{a} vs perfect play: {r}");
            return 0;
        }
        var ea = Load<TGame>(a, out int la);
        var eb = Load<TGame>(b, out int lb);
        int sa = a.StartsWith("rollout:") ? int.Parse(a[8..]) : sims;
        int sb = b.StartsWith("rollout:") ? int.Parse(b[8..]) : simsB;
        var result = Arena.Play(ea, eb, new ArenaConfig
        {
            Games = games, SimulationsA = sa, SimulationsB = sb, Parallel = Math.Min(games, 200), RandomOpeningMoves = openings,
            LeavesPerStepA = la, LeavesPerStepB = lb,
        }, r => { if (r.Games % 20 == 0) Console.Write($"\r{r.Games}/{games}: {r}   "); });
        Console.WriteLine($"\r{a} vs {b}: {result}                      ");
        return 0;
    }

    static IEvaluator<TGame> Load<TGame>(string spec, out int leaves) where TGame : unmanaged, IGame<TGame>
    {
        if (spec.StartsWith("rollout:"))
        {
            leaves = 16;
            return new ParallelRolloutEvaluator<TGame>(11);
        }
        leaves = 4;
        return Models.Load<TGame>(spec);
    }
}

/// <summary>gambit export --model runs/connect4/models/best.pt --out models/connect4.gnet</summary>
public sealed class ExportCommand : IGameCommand
{
    public int Run<TGame>(Args args) where TGame : unmanaged, IGame<TGame>
    {
        string model = args.Get("model", ""), output = args.Get("out", Path.Combine("models", $"{TGame.Name}.gnet"));
        args.CheckUnused();
        var net = Checkpoints.Load(model, cuda.is_available() ? CUDA : CPU);
        Checkpoints.Export(net, output);
        Console.WriteLine($"{model} ({net.Spec}, {net.ParameterCount:N0} parameters) -> {output} ({new FileInfo(output).Length / 1024} KB)");
        return 0;
    }
}

/// <summary>gambit play --game connect4 --model models/connect4.gnet [--sims 800 --second]: play in the console.</summary>
public sealed class PlayCommand : IGameCommand
{
    public int Run<TGame>(Args args) where TGame : unmanaged, IGame<TGame>
    {
        string model = args.Get("model", Path.Combine("models", $"{TGame.Name}.gnet"));
        int sims = args.Int("sims", 800);
        bool humanSecond = args.Flag("second");
        args.CheckUnused();
        var ai = new MctsPlayer<TGame>(Models.Load<TGame>(model), sims, 16);
        var g = TGame.Initial;
        while (!g.Outcome.IsOver())
        {
            Console.WriteLine();
            Console.Write(g.Render());
            if ((g.SideToMove == 1) == humanSecond)
            {
                Console.Write("your move: ");
                string? line = Console.ReadLine();
                if (line == null) return 0;
                int a = g.ParseAction(line.Trim());
                if (a < 0 || !g.IsLegal(a))
                {
                    Console.WriteLine("illegal move");
                    continue;
                }
                g.Play(a);
            }
            else
            {
                var r = ai.Search(g, sims);
                Console.WriteLine($"AI plays {g.ActionName(r.BestMove)} (value {r.RootValue:+0.00;-0.00}, " +
                                  $"line {string.Join(" ", r.PrincipalVariation.Select(g.ActionName))})");
                g.Play(r.BestMove);
            }
        }
        Console.Write(g.Render());
        Console.WriteLine(g.Outcome == Outcome.Draw ? "draw" : (((g.Ply - 1) & 1) == 1) == humanSecond ? "you win" : "the AI wins");
        return 0;
    }
}

/// <summary>gambit chart --game connect4 --run runs/connect4: redraws progress.svg from the run's log.</summary>
public sealed class ChartCommand : IGameCommand
{
    public int Run<TGame>(Args args) where TGame : unmanaged, IGame<TGame>
    {
        string run = args.Get("run", Path.Combine("runs", TGame.Name));
        string output = args.Get("out", Path.Combine(run, "progress.svg"));
        args.CheckUnused();
        var logs = File.ReadLines(Path.Combine(run, "log.jsonl"))
                       .Select(l => System.Text.Json.JsonSerializer.Deserialize<TrainCommand.GenerationLog>(l)!).ToList();
        var elo = EloTable.Load(Path.Combine(run, "elo.tsv"));
        string tpath = Path.Combine(run, "tournament.tsv");
        var tournament = File.Exists(tpath)
            ? File.ReadLines(tpath).Select(l => l.Split('	')).ToDictionary(f => f[0], f => double.Parse(f[1], System.Globalization.CultureInfo.InvariantCulture))
            : null;
        Charts.WriteProgress(output, TGame.Name, logs, elo.Compute(TrainCommand.Anchors[0].Name, 0), tournament);
        Console.WriteLine($"{output}: {logs.Count} generations");
        return 0;
    }
}

/// <summary>
/// gambit tournament --game connect4 --run runs/connect4 [--every 10 --games 100 --sims 200]
/// A round robin between saved generations (every N-th accepted network, plus the last) and the
/// rollout anchors; Elo by maximum likelihood over all the games, anchored at pure MCTS with 200
/// simulations = 0. Unlike the gating chain (each candidate only meets the previous best), every
/// network meets every other, so the ratings are directly comparable.
/// </summary>
public sealed class TournamentCommand : IGameCommand
{
    public int Run<TGame>(Args args) where TGame : unmanaged, IGame<TGame>
    {
        string run = args.Get("run", Path.Combine("runs", TGame.Name));
        int every = args.Int("every", 10), games = args.Int("games", 100), sims = args.Int("sims", 200);
        args.CheckUnused();
        var models = Directory.GetFiles(Path.Combine(run, "models"), "gen-*.pt")
            .Select(p => (Path: p, Gen: int.Parse(Path.GetFileNameWithoutExtension(p)[4..])))
            .OrderBy(m => m.Gen).ToList();
        var chosen = new List<(string Path, int Gen)>();
        foreach (var m in models)
            if (chosen.Count == 0 || m.Gen >= chosen[^1].Gen + every) chosen.Add(m);
        if (chosen[^1].Gen != models[^1].Gen) chosen.Add(models[^1]);
        Console.WriteLine($"players: {string.Join(", ", chosen.Select(c => $"gen-{c.Gen:D4}"))} + rollout anchors");

        var players = chosen.Select(c => (Name: $"gen-{c.Gen:D4}", Eval: Models.Load<TGame>(c.Path), Sims: sims, Leaves: 4)).ToList();
        foreach (var (name, n) in TrainCommand.Anchors)
            players.Add((name, new ParallelRolloutEvaluator<TGame>(5), n, 16));
        var elo = new EloTable();
        for (int i = 0; i < players.Count; i++)
            for (int j = i + 1; j < players.Count; j++)
            {
                var (a, b) = (players[i], players[j]);
                if (a.Name.StartsWith("mcts") && b.Name.StartsWith("mcts")) continue;
                var r = Arena.Play(a.Eval, b.Eval, new ArenaConfig
                {
                    Games = games, SimulationsA = a.Sims, SimulationsB = b.Sims, Parallel = games, Seed = i * 100 + j,
                    LeavesPerStepA = a.Leaves, LeavesPerStepB = b.Leaves,
                });
                elo.Add(a.Name, b.Name, r.WinsA, r.Draws, r.WinsB);
                Console.WriteLine($"{a.Name} vs {b.Name}: {r}");
            }
        var ratings = elo.Compute(TrainCommand.Anchors[0].Name, 0);
        string path = Path.Combine(run, "tournament.tsv");
        File.WriteAllLines(path, ratings.OrderBy(kv => kv.Value).Select(kv => $"{kv.Key}\t{kv.Value:F0}"));
        elo.Save(Path.Combine(run, "tournament-games.tsv"));
        Console.WriteLine();
        foreach (var (name, r) in ratings.OrderBy(kv => kv.Value)) Console.WriteLine($"{name,-20} {r,6:F0}");
        return 0;
    }
}
