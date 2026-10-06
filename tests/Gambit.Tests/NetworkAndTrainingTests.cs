using Gambit.Core.Games;
using Gambit.Core.Neural;
using Gambit.Core.Rating;
using Gambit.Core.Search;
using Gambit.Core.Training;

namespace Gambit.Tests;

public class NativeNetworkTests
{
    /// <summary>Random weights for a small network, in the exported (BN-folded, PyTorch-layout) form.</summary>
    static (NetworkSpec Spec, List<float[]> Weights) RandomNetwork(int channels, int seed)
    {
        var spec = NetworkSpec.For<ConnectFour>(2, channels) with { PolicyChannels = 2, ValueChannels = 1, ValueHidden = 16 };
        var rng = new Random(seed);
        float[] R(int n, float scale) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1) * scale).ToArray();
        int c = spec.Channels, area = spec.Area;
        var w = new List<float[]> { R(c * spec.Planes * 9, 0.3f), R(c, 0.1f) };
        for (int b = 0; b < spec.Blocks; b++)
        {
            w.Add(R(c * c * 9, 0.08f)); w.Add(R(c, 0.1f));
            w.Add(R(c * c * 9, 0.08f)); w.Add(R(c, 0.1f));
        }
        w.Add(R(spec.PolicyChannels * c, 0.2f)); w.Add(R(spec.PolicyChannels, 0.1f));
        w.Add(R(spec.Actions * spec.PolicyChannels * area, 0.2f)); w.Add(R(spec.Actions, 0.1f));
        w.Add(R(spec.ValueChannels * c, 0.2f)); w.Add(R(spec.ValueChannels, 0.1f));
        w.Add(R(spec.ValueHidden * spec.ValueChannels * area, 0.2f)); w.Add(R(spec.ValueHidden, 0.1f));
        w.Add(R(spec.ValueHidden, 0.2f)); w.Add(R(1, 0.1f));
        return (spec, w);
    }

    /// <summary>The same network computed in the most direct way: CHW tensors, 4 nested loops per convolution.</summary>
    static (float[] Logits, float Value) Reference(NetworkSpec s, List<float[]> w, float[] input)
    {
        int H = s.Height, W = s.Width, area = H * W, k = 0;
        float[] Conv(float[] x, int cin, float[] wt, float[] b, bool relu)
        {
            int cout = b.Length;
            var y = new float[cout * area];
            for (int co = 0; co < cout; co++)
                for (int yy = 0; yy < H; yy++)
                    for (int xx = 0; xx < W; xx++)
                    {
                        double acc = b[co];
                        for (int ci = 0; ci < cin; ci++)
                            for (int ky = 0; ky < 3; ky++)
                                for (int kx = 0; kx < 3; kx++)
                                {
                                    int sy = yy + ky - 1, sx = xx + kx - 1;
                                    if (sy < 0 || sx < 0 || sy >= H || sx >= W) continue;
                                    acc += x[ci * area + sy * W + sx] * wt[((co * cin + ci) * 3 + ky) * 3 + kx];
                                }
                        y[co * area + yy * W + xx] = relu ? (float)Math.Max(0, acc) : (float)acc;
                    }
            return y;
        }
        float[] Conv1(float[] x, float[] wt, float[] b)
        {
            var y = new float[b.Length * area];
            for (int co = 0; co < b.Length; co++)
                for (int p = 0; p < area; p++)
                {
                    double acc = b[co];
                    for (int ci = 0; ci < s.Channels; ci++) acc += x[ci * area + p] * wt[co * s.Channels + ci];
                    y[co * area + p] = (float)Math.Max(0, acc);
                }
            return y;
        }
        float[] Fc(float[] x, float[] wt, float[] b, bool relu)
        {
            var y = new float[b.Length];
            for (int o = 0; o < b.Length; o++)
            {
                double acc = b[o];
                for (int i = 0; i < x.Length; i++) acc += x[i] * wt[o * x.Length + i];
                y[o] = relu ? (float)Math.Max(0, acc) : (float)acc;
            }
            return y;
        }
        var h = Conv(input, s.Planes, w[k++], w[k++], true);
        for (int b = 0; b < s.Blocks; b++)
        {
            var t = Conv(h, s.Channels, w[k++], w[k++], true);
            var u = Conv(t, s.Channels, w[k++], w[k++], false);
            for (int i = 0; i < h.Length; i++) h[i] = Math.Max(0, h[i] + u[i]);
        }
        var pol = Fc(Conv1(h, w[k++], w[k++]), w[k++], w[k++], false);
        var val = Fc(Fc(Conv1(h, w[k++], w[k++]), w[k++], w[k++], true), w[k++], w[k++], false);
        return (pol, MathF.Tanh(val[0]));
    }

    [Theory]
    [InlineData(64)]    // the AVX2 path (64 output channels at a time)
    [InlineData(24)]    // the generic path
    public void Native_inference_matches_a_direct_implementation(int channels)
    {
        var (spec, w) = RandomNetwork(channels, channels);
        var net = new NativeNetwork(spec, w);
        var g = ConnectFour.FromMoves("4453123");
        var planes = new float[spec.Planes * spec.Area];
        g.Encode(planes);
        var logits = new float[7];
        float v = net.Evaluate(planes, logits);
        var (refLogits, refValue) = Reference(spec, w, planes);
        for (int i = 0; i < 7; i++) Assert.Equal(refLogits[i], logits[i], 3e-4f);
        Assert.Equal(refValue, v, 3e-5f);
    }

    [Fact]
    public void Gnet_files_round_trip()
    {
        var (spec, w) = RandomNetwork(16, 3);
        string path = Path.GetTempFileName();
        try
        {
            NativeNetwork.Write(path, spec, w);
            var a = new NativeNetwork(spec, w).For<ConnectFour>();
            var b = NativeNetwork.Load(path).For<ConnectFour>();
            var s = new[] { ConnectFour.FromMoves("44"), ConnectFour.FromMoves("1234567") };
            var la = new float[14]; var va = new float[2];
            var lb = new float[14]; var vb = new float[2];
            a.Evaluate(s, la, va);
            b.Evaluate(s, lb, vb);
            Assert.Equal(la, lb);
            Assert.Equal(va, vb);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Wrong_game_is_rejected()
    {
        var (spec, w) = RandomNetwork(16, 4);
        Assert.Throws<InvalidOperationException>(() => new NativeNetwork(spec, w).For<Gomoku>());
    }
}

public class EloTests
{
    [Fact]
    public void Recovers_the_rating_gap_implied_by_a_score()
    {
        var t = new EloTable();
        t.Add("a", "b", 760, 0, 240);    // 76% -> about +200
        var r = t.Compute("b");
        Assert.InRange(r["a"], 190, 210);
        Assert.Equal(0, r["b"], 6);
    }

    [Fact]
    public void Chains_ratings_through_common_opponents()
    {
        var t = new EloTable();
        t.Add("a", "b", 640, 0, 360);    // a ~ b + 100
        t.Add("b", "c", 640, 0, 360);    // b ~ c + 100
        var r = t.Compute("c");
        Assert.InRange(r["b"], 90, 110);
        Assert.InRange(r["a"], 180, 220);
    }

    [Fact]
    public void A_perfect_score_stays_finite_and_tables_round_trip()
    {
        var t = new EloTable();
        t.Add("strong", "weak", 50, 0, 0);
        var r = t.Compute("weak");
        Assert.True(double.IsFinite(r["strong"]) && r["strong"] > 300);
        string path = Path.GetTempFileName();
        try
        {
            t.Save(path);
            Assert.Equal(r["strong"], EloTable.Load(path).Compute("weak")["strong"], 6);
        }
        finally
        {
            File.Delete(path);
        }
    }
}

public class SelfPlayAndArenaTests
{
    [Fact]
    public void Self_play_produces_complete_games_with_valid_targets()
    {
        var engine = new SelfPlayEngine<ConnectFour>(new RolloutEvaluator<ConnectFour>(1, 3), new SelfPlayConfig
        {
            ParallelGames = 8, Simulations = 40, FastSimulations = 10, FullSearchProbability = 0.5f, Seed = 3,
        });
        var stats = new SelfPlayStats();
        var games = engine.Play(12, stats);
        Assert.Equal(12, games.Count);
        Assert.Equal(12, stats.Games);
        foreach (var g in games)
        {
            var state = ConnectFour.Initial;
            foreach (int m in g.Moves) state.Play(m);
            Assert.True(state.Outcome.IsOver());
            foreach (var s in g.Samples)
            {
                Assert.Equal(1f, s.Policy.Sum(), 4);
                Assert.All(Enumerable.Range(0, 7), a => Assert.True(s.Policy[a] == 0 || s.State.IsLegal(a)));
                Assert.Contains(s.ValueTarget, new[] { -1f, 0f, 1f });
            }
            // The value target is the final result seen from each sample's side to move.
            foreach (var s in g.Samples)
            {
                float expected = g.Result == 0 ? 0 : (s.State.SideToMove == 0 ? g.Result : -g.Result);
                Assert.Equal(expected, s.ValueTarget);
            }
        }
    }

    [Fact]
    public void Replay_files_round_trip_and_symmetry_preserves_targets()
    {
        var engine = new SelfPlayEngine<ConnectFour>(new UniformEvaluator<ConnectFour>(), new SelfPlayConfig
        {
            ParallelGames = 4, Simulations = 16, FastSimulations = 16, FullSearchProbability = 1f, Seed = 9,
        });
        var games = engine.Play(4, new SelfPlayStats());
        string path = Path.GetTempFileName();
        try
        {
            ReplayBuffer<ConnectFour>.Save(path, games);
            var loaded = ReplayBuffer<ConnectFour>.Load(path);
            Assert.Equal(games.Count, loaded.Count);
            for (int i = 0; i < games.Count; i++)
            {
                Assert.Equal(games[i].Moves, loaded[i].Moves);
                Assert.Equal(games[i].Samples.Count, loaded[i].Samples.Count);
                for (int k = 0; k < games[i].Samples.Count; k++)
                {
                    Assert.Equal(games[i].Samples[k].State.Hash, loaded[i].Samples[k].State.Hash);
                    Assert.Equal(games[i].Samples[k].Policy, loaded[i].Samples[k].Policy);
                }
            }
        }
        finally
        {
            File.Delete(path);
        }

        // The mirrored encoding equals the encoding of the mirrored position.
        var g = ConnectFour.FromMoves("1123");
        var m = ConnectFour.FromMoves("7765");
        var a = new float[126]; var b = new float[126]; var tmp = new float[126];
        Symmetry.Encode(g, 1, a, tmp);
        m.Encode(b);
        Assert.Equal(b, a);
    }

    [Fact]
    public void Arena_counts_results_and_alternates_colours()
    {
        // A searcher with rollouts against a searcher that knows nothing (uniform, value 0).
        var r = Arena.Play<ConnectFour>(new RolloutEvaluator<ConnectFour>(2, 1), new UniformEvaluator<ConnectFour>(),
                                        new ArenaConfig { Games = 20, SimulationsA = 100, SimulationsB = 100, Parallel = 10, RandomOpeningMoves = 2 });
        Assert.Equal(20, r.Games);
        Assert.True(r.ScoreA > 0.7, $"rollouts should beat a blind search, got {r}");
    }

    [Fact]
    public void Gomoku_search_completes_five()
    {
        // Black has four in a row on the bottom row (a1-d1): e1 completes five, and the search must find it.
        var g = Gomoku.Initial;
        foreach (var a in new[] { 72, 0, 73, 1, 74, 2, 75, 4 }) g.Play(a);
        var player = new MctsPlayer<Gomoku>(new RolloutEvaluator<Gomoku>(1, 4), 400);
        int move = player.ChooseMove(g);
        Assert.Equal(76, move);
    }
}

public class OpeningAndGatingTests
{
    [Fact]
    public void Likelihood_of_superiority_ignores_draws()
    {
        Assert.Equal(0.5, new MatchResult(10, 80, 10).LikelihoodOfSuperiority, 6);
        Assert.InRange(new MatchResult(20, 170, 10).LikelihoodOfSuperiority, 0.95, 0.97);   // Phi(10 / sqrt(30)) = 0.966
        Assert.True(new MatchResult(60, 0, 40).LikelihoodOfSuperiority < new MatchResult(30, 160, 10).LikelihoodOfSuperiority);
    }

    [Fact]
    public void Random_openings_make_self_play_games_differ()
    {
        // A deterministic evaluator: without random openings every game would be identical.
        var engine = new SelfPlayEngine<ConnectFour>(new UniformEvaluator<ConnectFour>(), new SelfPlayConfig
        {
            ParallelGames = 16, Simulations = 8, FastSimulations = 8, FullSearchProbability = 1f, TemperatureMoves = 0,
            RandomOpeningMoves = 4, Seed = 5,
        });
        var games = engine.Play(16, new SelfPlayStats());
        int distinct = games.Select(g => string.Join(",", g.Moves)).Distinct().Count();
        Assert.True(distinct >= 10, $"only {distinct} distinct games");
        foreach (var g in games)
        {
            var s = ConnectFour.Initial;
            foreach (int m in g.Moves) s.Play(m);
            Assert.True(s.Outcome.IsOver());
        }
    }

    [Fact]
    public void Identical_players_split_paired_openings_evenly()
    {
        // Equally strong players, every opening played once from each side: the score must be near 50%.
        var r = Arena.Play<ConnectFour>(new RolloutEvaluator<ConnectFour>(1, 1), new RolloutEvaluator<ConnectFour>(1, 1),
                                        new ArenaConfig { Games = 40, SimulationsA = 30, SimulationsB = 30, Parallel = 40, RandomOpeningMoves = 3 });
        Assert.Equal(40, r.Games);
        Assert.InRange(r.ScoreA, 0.3, 0.7);
    }
}
