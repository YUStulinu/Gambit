using Gambit.Core.Games;
using Gambit.Core.Training;
using TorchSharp;
using static TorchSharp.torch;

namespace Gambit.Neural;

public sealed record TrainConfig
{
    public int BatchSize { get; init; } = 512;
    public double LearningRate { get; init; } = 0.02;
    public double Momentum { get; init; } = 0.9;
    public double WeightDecay { get; init; } = 1e-4;
    public float ValueLossWeight { get; init; } = 1.0f;
}

public sealed record TrainMetrics(double PolicyLoss, double ValueLoss, double PolicyEntropy, int Steps, double Seconds)
{
    public double TotalLoss => PolicyLoss + ValueLoss;
    public override string ToString() =>
        $"policy loss {PolicyLoss:F4}, value loss {ValueLoss:F4}, target entropy {PolicyEntropy:F3}, {Steps} steps in {Seconds:F1}s";
}

/// <summary>
/// Trains the network on replay-buffer batches:
///     loss = cross-entropy(policy logits, search visit distribution) + w * (value - target)^2
/// with SGD + Nesterov momentum and L2 weight decay, as in AlphaZero.
/// </summary>
public sealed class Trainer<TGame> where TGame : unmanaged, IGame<TGame>
{
    readonly AlphaZeroNet _net;
    readonly Device _device;
    readonly TrainConfig _cfg;
    readonly optim.Optimizer _opt;
    readonly float[] _planes, _policies, _values;
    readonly Random _rng;

    public AlphaZeroNet Network => _net;
    public double LearningRate { get; set; }

    public Trainer(AlphaZeroNet net, Device device, TrainConfig cfg, int seed = 0)
    {
        _net = net;
        _device = device;
        _cfg = cfg;
        LearningRate = cfg.LearningRate;
        _opt = optim.SGD(net.parameters(), cfg.LearningRate, momentum: cfg.Momentum, nesterov: true,
                         weight_decay: cfg.WeightDecay);
        int b = cfg.BatchSize;
        _planes = new float[b * TGame.PlaneCount * TGame.Height * TGame.Width];
        _policies = new float[b * TGame.ActionCount];
        _values = new float[b];
        _rng = new Random(seed);
    }

    /// <summary>Runs <paramref name="steps"/> optimisation steps on random batches from <paramref name="buffer"/>.</summary>
    public TrainMetrics Train(ReplayBuffer<TGame> buffer, int steps)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        foreach (var g in _opt.ParamGroups) g.LearningRate = LearningRate;
        _net.train();
        int b = _cfg.BatchSize;
        double pl = 0, vl = 0, ent = 0;
        for (int step = 0; step < steps; step++)
        {
            buffer.FillBatch(_rng, b, _planes, _policies, _values);
            using var scope = NewDisposeScope();
            var x = tensor(_planes, [b, TGame.PlaneCount, TGame.Height, TGame.Width]).to(_device);
            var pi = tensor(_policies, [b, TGame.ActionCount]).to(_device);
            var z = tensor(_values, [b]).to(_device);
            var (logits, v) = _net.forward(x);
            var policyLoss = -(pi * nn.functional.log_softmax(logits, 1)).sum(1).mean();
            var valueLoss = (v - z).pow(2).mean();
            var loss = policyLoss + _cfg.ValueLossWeight * valueLoss;
            _opt.zero_grad();
            loss.backward();
            _opt.step();
            if (step == steps - 1 || step % 50 == 0)
            {
                // Entropy of the targets: the floor of the policy loss (cross-entropy = entropy + KL).
                var targetEntropy = -(pi * (pi + 1e-12f).log()).sum(1).mean();
                pl = policyLoss.item<float>();
                vl = valueLoss.item<float>();
                ent = targetEntropy.item<float>();
            }
        }
        _net.eval();
        return new TrainMetrics(pl, vl, ent, steps, sw.Elapsed.TotalSeconds);
    }

    /// <summary>Mean losses over a fixed set of held-out samples (no augmentation, no gradient).</summary>
    public (double Policy, double Value) Evaluate(ReplayBuffer<TGame> data, int maxSamples = 4096)
    {
        _net.eval();
        int n = Math.Min(maxSamples, data.Count);
        if (n == 0) return (double.NaN, double.NaN);
        int ps = TGame.PlaneCount * TGame.Height * TGame.Width, a = TGame.ActionCount;
        var planes = new float[n * ps];
        var pol = new float[n * a];
        var val = new float[n];
        for (int i = 0; i < n; i++)
        {
            var s = data[i];
            s.State.Encode(planes.AsSpan(i * ps, ps));
            s.Policy.CopyTo(pol, i * a);
            val[i] = s.ValueTarget;
        }
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        var (logits, v) = _net.forward(tensor(planes, [n, TGame.PlaneCount, TGame.Height, TGame.Width]).to(_device));
        var pi = tensor(pol, [n, a]).to(_device);
        double policyLoss = (-(pi * nn.functional.log_softmax(logits, 1)).sum(1).mean()).item<float>();
        double valueLoss = (v - tensor(val).to(_device)).pow(2).mean().item<float>();
        return (policyLoss, valueLoss);
    }
}
