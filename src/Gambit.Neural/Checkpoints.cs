using Gambit.Core.Neural;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Gambit.Neural;

/// <summary>Saving, loading, copying and exporting networks.</summary>
public static class Checkpoints
{
    /// <summary>Saves the weights (TorchSharp format) with the spec in a sidecar JSON file.</summary>
    public static void Save(AlphaZeroNet net, string path)
    {
        net.save(path);
        File.WriteAllText(path + ".json", System.Text.Json.JsonSerializer.Serialize(net.Spec));
    }

    public static AlphaZeroNet Load(string path, Device device)
    {
        var spec = System.Text.Json.JsonSerializer.Deserialize<NetworkSpec>(File.ReadAllText(path + ".json"))
                   ?? throw new InvalidDataException($"cannot read {path}.json");
        var net = new AlphaZeroNet(spec);
        net.load(path);
        return net.to(device);
    }

    /// <summary>An independent copy of the network (same weights and batch-norm statistics) on <paramref name="device"/>.</summary>
    public static AlphaZeroNet Clone(AlphaZeroNet source, Device device)
    {
        var copy = new AlphaZeroNet(source.Spec);
        using (no_grad())
        {
            var dst = copy.state_dict();
            foreach (var (name, t) in source.state_dict())
                dst[name].copy_(t.to(dst[name].device));
        }
        return copy.to(device);
    }

    static float[] ToArray(Tensor t) => t.detach().to(ScalarType.Float32).cpu().contiguous().data<float>().ToArray();

    /// <summary>Folds a batch norm into the convolution before it: returns (weights, bias).</summary>
    static (float[] W, float[] B) Fold(Conv2d conv, BatchNorm2d bn)
    {
        using var scope = NewDisposeScope();
        var w = conv.weight!.detach().to(ScalarType.Float32).cpu();
        var scale = bn.weight!.detach().cpu() / (bn.running_var!.detach().cpu() + bn.eps).sqrt();
        var folded = w * scale.reshape(-1, 1, 1, 1);
        var bias = bn.bias!.detach().cpu() - bn.running_mean!.detach().cpu() * scale;
        return (ToArray(folded.reshape(w.shape[0], -1)), ToArray(bias));
    }

    /// <summary>The weight arrays in the order <see cref="NativeNetwork"/> expects (batch norms folded).</summary>
    public static List<float[]> NativeWeights(AlphaZeroNet net)
    {
        var list = new List<float[]>();
        void Add((float[] W, float[] B) wb)
        {
            list.Add(wb.W);
            list.Add(wb.B);
        }
        Add(Fold(net.StemConv, net.StemBn));
        foreach (var block in net.Tower)
        {
            Add(Fold(block.Conv1, block.Bn1));
            Add(Fold(block.Conv2, block.Bn2));
        }
        Add(Fold(net.PolicyConv, net.PolicyBn));
        list.Add(ToArray(net.PolicyFc.weight!));
        list.Add(ToArray(net.PolicyFc.bias!));
        Add(Fold(net.ValueConv, net.ValueBn));
        list.Add(ToArray(net.ValueFc1.weight!));
        list.Add(ToArray(net.ValueFc1.bias!));
        list.Add(ToArray(net.ValueFc2.weight!));
        list.Add(ToArray(net.ValueFc2.bias!));
        return list;
    }

    /// <summary>Writes the network as a .gnet file for the pure C# inference engine.</summary>
    public static void Export(AlphaZeroNet net, string path) => NativeNetwork.Write(path, net.Spec, NativeWeights(net));

    public static NativeNetwork ToNative(AlphaZeroNet net) => new(net.Spec, NativeWeights(net));
}
