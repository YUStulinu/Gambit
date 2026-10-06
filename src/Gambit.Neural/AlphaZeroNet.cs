using Gambit.Core.Neural;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace Gambit.Neural;

/// <summary>conv3x3 - BN - ReLU - conv3x3 - BN, plus the input, then ReLU.</summary>
public sealed class ResidualBlock : Module<Tensor, Tensor>
{
    public readonly Conv2d Conv1, Conv2;
    public readonly BatchNorm2d Bn1, Bn2;

    public ResidualBlock(int channels) : base("block")
    {
        Conv1 = Conv2d(channels, channels, 3, padding: 1, bias: false);
        Bn1 = BatchNorm2d(channels);
        Conv2 = Conv2d(channels, channels, 3, padding: 1, bias: false);
        Bn2 = BatchNorm2d(channels);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        using var h = functional.relu(Bn1.forward(Conv1.forward(x)));
        using var y = Bn2.forward(Conv2.forward(h));
        return functional.relu(y.add_(x));
    }
}

/// <summary>The AlphaZero two-headed residual network (see <see cref="NetworkSpec"/>).</summary>
public sealed class AlphaZeroNet : Module<Tensor, (Tensor Policy, Tensor Value)>
{
    public readonly NetworkSpec Spec;
    public readonly Conv2d StemConv;
    public readonly BatchNorm2d StemBn;
    public readonly ModuleList<ResidualBlock> Tower;
    public readonly Conv2d PolicyConv;
    public readonly BatchNorm2d PolicyBn;
    public readonly Linear PolicyFc;
    public readonly Conv2d ValueConv;
    public readonly BatchNorm2d ValueBn;
    public readonly Linear ValueFc1, ValueFc2;

    public AlphaZeroNet(NetworkSpec spec) : base("alphazero")
    {
        Spec = spec;
        int c = spec.Channels;
        StemConv = Conv2d(spec.Planes, c, 3, padding: 1, bias: false);
        StemBn = BatchNorm2d(c);
        Tower = new ModuleList<ResidualBlock>();
        for (int i = 0; i < spec.Blocks; i++) Tower.Add(new ResidualBlock(c));
        PolicyConv = Conv2d(c, spec.PolicyChannels, 1, bias: false);
        PolicyBn = BatchNorm2d(spec.PolicyChannels);
        PolicyFc = Linear(spec.PolicyChannels * spec.Area, spec.Actions);
        ValueConv = Conv2d(c, spec.ValueChannels, 1, bias: false);
        ValueBn = BatchNorm2d(spec.ValueChannels);
        ValueFc1 = Linear(spec.ValueChannels * spec.Area, spec.ValueHidden);
        ValueFc2 = Linear(spec.ValueHidden, 1);
        RegisterComponents();
    }

    public override (Tensor Policy, Tensor Value) forward(Tensor x)
    {
        using var scope = NewDisposeScope();
        var h = functional.relu(StemBn.forward(StemConv.forward(x)));
        foreach (var block in Tower) h = block.forward(h);
        var p = functional.relu(PolicyBn.forward(PolicyConv.forward(h))).flatten(1);
        var policy = PolicyFc.forward(p);
        var v = functional.relu(ValueBn.forward(ValueConv.forward(h))).flatten(1);
        var value = ValueFc2.forward(functional.relu(ValueFc1.forward(v))).tanh().squeeze(1);
        return (policy.MoveToOuterDisposeScope(), value.MoveToOuterDisposeScope());
    }

    public long ParameterCount => parameters().Sum(p => p.numel());
}
