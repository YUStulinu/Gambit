using Gambit.Core.Games;
using Gambit.Core.Neural;
using Gambit.Core.Search;
using TorchSharp;
using static TorchSharp.torch;

namespace Gambit.Neural;

/// <summary>
/// Evaluates positions with the network on the GPU: the batch is encoded on the CPU (in parallel
/// for large batches), copied to the device in one transfer, run through the network under
/// no_grad, and the logits and values copied back. Optionally in fp16, which on tensor-core GPUs
/// roughly doubles throughput at no measurable cost in playing strength.
/// </summary>
public sealed class TorchEvaluator<TGame> : IEvaluator<TGame>, IDisposable where TGame : struct, IGame<TGame>
{
    readonly AlphaZeroNet _net;
    readonly Device _device;
    readonly bool _half;
    float[] _input = [];
    readonly int _planeSize = TGame.PlaneCount * TGame.Height * TGame.Width;

    public long Positions { get; private set; }
    public int BatchBucket { get; set; } = 128;
    /// <summary>Time spent encoding positions, and in the network call (transfer + forward + copy back).</summary>
    public readonly System.Diagnostics.Stopwatch EncodeTime = new(), NetworkTime = new();

    /// <summary>Wraps a copy of <paramref name="source"/>'s weights for inference.</summary>
    public TorchEvaluator(AlphaZeroNet source, Device device, bool half)
    {
        _device = device;
        _half = half && device.type == DeviceType.CUDA;
        _net = Checkpoints.Clone(source, device);
        _net.eval();
        if (_half) _net.half();
    }

    public NetworkSpec Spec => _net.Spec;

    public void Evaluate(ReadOnlySpan<TGame> states, Span<float> logits, Span<float> values)
    {
        int n = states.Length, a = TGame.ActionCount, ps = _planeSize;
        // Pad the batch to a multiple of BatchBucket: cuDNN plans its convolutions per input shape,
        // and with a new shape at every call the planning costs more than the convolutions.
        int padded = n <= 8 ? n : (n + BatchBucket - 1) / BatchBucket * BatchBucket;
        EncodeTime.Start();
        if (_input.Length < padded * ps) _input = new float[padded * ps];
        _input.AsSpan(n * ps, (padded - n) * ps).Clear();
        if (n >= 64)
        {
            var arr = states.ToArray();
            var input = _input;
            Parallel.For(0, (n + 63) / 64, chunk =>
            {
                for (int i = chunk * 64; i < Math.Min(n, chunk * 64 + 64); i++) arr[i].Encode(input.AsSpan(i * ps, ps));
            });
        }
        else
        {
            for (int i = 0; i < n; i++) states[i].Encode(_input.AsSpan(i * ps, ps));
        }

        EncodeTime.Stop();
        NetworkTime.Start();
        using var scope = NewDisposeScope();
        using var noGrad = no_grad();
        var x = tensor(_input.AsSpan(0, padded * ps).ToArray(), [padded, TGame.PlaneCount, TGame.Height, TGame.Width]).to(_device);
        if (_half) x = x.to(ScalarType.Float16);
        var (p, v) = _net.forward(x);
        var pc = p.to(ScalarType.Float32).cpu();
        var vc = v.to(ScalarType.Float32).cpu();
        var pl = pc.data<float>().ToArray();
        var vl = vc.data<float>().ToArray();
        pl.AsSpan(0, n * a).CopyTo(logits);
        vl.AsSpan(0, n).CopyTo(values);
        Positions += n;
        NetworkTime.Stop();
    }

    public void Dispose() => _net.Dispose();
}
