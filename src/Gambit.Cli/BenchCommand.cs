using System.Diagnostics;
using Gambit.Core.Games;
using Gambit.Core.Neural;
using Gambit.Core.Search;
using Gambit.Core.Solver;
using Gambit.Core.Training;
using Gambit.Neural;
using TorchSharp;
using static TorchSharp.torch;

namespace Gambit.Cli;

/// <summary>
/// gambit bench --game connect4 [--blocks 6 --channels 64]
/// Measures the pieces that decide training speed: network throughput on the GPU (fp32 and
/// fp16, several batch sizes), the pure C# network, the move generator, the solver, and
/// end-to-end self-play.
/// </summary>
public sealed class BenchCommand : IGameCommand
{
    public int Run<TGame>(Args args) where TGame : unmanaged, IGame<TGame>
    {
        int blocks = args.Int("blocks", 6), channels = args.Int("channels", 64);
        int selfPlayGames = args.Int("games", 512);
        int sims = args.Int("sims", 200);
        bool skipSelfPlay = args.Flag("no-selfplay");
        var device = cuda.is_available() ? CUDA : CPU;
        string? modelArg = args.GetOrNull("model");
        args.CheckUnused();
        Console.WriteLine($"{TGame.Name}: network {blocks}x{channels}, device {device.type}");

        // Raw move generation speed: random playouts.
        var rng = new Random(1);
        Span<int> moves = stackalloc int[TGame.ActionCount];
        var sw = Stopwatch.StartNew();
        long plies = 0;
        int playouts = 0;
        while (sw.ElapsedMilliseconds < 1000)
        {
            var g = TGame.Initial;
            while (!g.Outcome.IsOver())
            {
                int n = g.LegalMoves(moves);
                g.Play(moves[rng.Next(n)]);
                plies++;
            }
            playouts++;
        }
        Console.WriteLine($"random playouts: {playouts / sw.Elapsed.TotalSeconds / 1e6:F2}M games/s, {plies / sw.Elapsed.TotalSeconds / 1e6:F1}M moves/s (one thread)");

        var net = modelArg != null ? Checkpoints.Load(modelArg, device) : new AlphaZeroNet(NetworkSpec.For<TGame>(blocks, channels)).to(device);
        Console.WriteLine($"parameters: {net.ParameterCount:N0}");
        var states = new TGame[1024];
        for (int i = 0; i < states.Length; i++)
        {
            var g = TGame.Initial;
            for (int k = 0; k < 6 && !g.Outcome.IsOver(); k++)
            {
                int n = g.LegalMoves(moves);
                g.Play(moves[rng.Next(n)]);
            }
            states[i] = g;
        }
        var logits = new float[1024 * TGame.ActionCount];
        var values = new float[1024];
        foreach (bool half in new[] { false, true })
        {
            using var ev = new TorchEvaluator<TGame>(net, device, half);
            foreach (int batch in new[] { 1, 64, 256, 512, 1024 })
            {
                for (int w = 0; w < 3; w++) ev.Evaluate(states.AsSpan(0, batch), logits, values);
                sw.Restart();
                int reps = 0;
                while (sw.ElapsedMilliseconds < 700)
                {
                    ev.Evaluate(states.AsSpan(0, batch), logits, values);
                    reps++;
                }
                Console.WriteLine($"GPU {(half ? "fp16" : "fp32")} batch {batch,4}: {reps * batch / sw.Elapsed.TotalSeconds,9:N0} positions/s " +
                                  $"({1000 * sw.Elapsed.TotalSeconds / reps:F2} ms/batch)");
            }
        }

        var native = Checkpoints.ToNative(net).For<TGame>();
        {
            // The C# network must reproduce the GPU network (fp32) on the same positions.
            using var ev32 = new TorchEvaluator<TGame>(net, device, false);
            int n = 256, act = TGame.ActionCount;
            var l1 = new float[n * act]; var v1 = new float[n];
            var l2 = new float[n * act]; var v2 = new float[n];
            ev32.Evaluate(states.AsSpan(0, n), l1, v1);
            native.Evaluate(states.AsSpan(0, n), l2, v2);
            double dl = 0, dv = 0;
            for (int i = 0; i < n * act; i++) dl = Math.Max(dl, Math.Abs(l1[i] - l2[i]));
            for (int i = 0; i < n; i++) dv = Math.Max(dv, Math.Abs(v1[i] - v2[i]));
            Console.WriteLine($"C# network vs GPU network (fp32): max |diff| logits {dl:E1}, values {dv:E1}");
        }
        sw.Restart();
        int count = 0;
        while (sw.ElapsedMilliseconds < 1000)
        {
            native.Evaluate(states.AsSpan(count % 1000, 1), logits, values);
            count++;
        }
        Console.WriteLine($"C# network, one thread: {count / sw.Elapsed.TotalSeconds:N0} positions/s");
        sw.Restart();
        native.Evaluate(states.AsSpan(0, 1024), logits, values);
        Console.WriteLine($"C# network, batch 1024 on all cores: {1024 / sw.Elapsed.TotalSeconds:N0} positions/s");

        if (typeof(TGame) == typeof(ConnectFour))
        {
            var solver = new ConnectFourSolver();
            foreach (int ply in new[] { 20, 14, 10, 8 })
            {
                sw.Restart();
                long nodes = 0;
                int solved = 0;
                var r = new Random(ply);
                while (sw.ElapsedMilliseconds < 3000 && solved < 50)
                {
                    var g = ConnectFour.Initial;
                    while (g.Ply < ply)
                    {
                        int n = g.LegalMoves(moves);
                        g.Play(moves[r.Next(n)]);
                        if (g.Outcome.IsOver()) g = ConnectFour.Initial;
                    }
                    solver.Solve(g);
                    nodes += solver.NodeCount;
                    solved++;
                }
                Console.WriteLine($"solver, positions after {ply,2} moves: {1000 * sw.Elapsed.TotalSeconds / solved,8:F1} ms each, {nodes / sw.Elapsed.TotalSeconds / 1e6:F1}M nodes/s");
            }
        }

        // End-to-end self-play with the (untrained) network.
        if (skipSelfPlay) return 0;
        foreach (bool half in new[] { false, true })
        {
            using var ev = new TorchEvaluator<TGame>(net, device, half);
            var cached = new CachedEvaluator<TGame>(ev);
            var engine = new SelfPlayEngine<TGame>(cached, new SelfPlayConfig
            {
                ParallelGames = selfPlayGames, Simulations = sims, FastSimulations = sims / 5, FullSearchProbability = 0.25f,
            });
            var stats = new SelfPlayStats();
            engine.Play(selfPlayGames, stats);
            Console.WriteLine($"self-play ({(half ? "fp16" : "fp32")}, {sims}/{sims / 5} sims, {selfPlayGames} parallel): {stats}, cache hits {100 * cached.HitRate:F0}%");
            Console.WriteLine($"    inside evaluate: encode {ev.EncodeTime.Elapsed.TotalSeconds / stats.Seconds * 100:F0}%, network {ev.NetworkTime.Elapsed.TotalSeconds / stats.Seconds * 100:F0}%");
        }
        return 0;
    }
}
