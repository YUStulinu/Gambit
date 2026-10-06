using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Gambit.Core.Games;
using Gambit.Core.Search;

namespace Gambit.Core.Neural;

/// <summary>
/// The trained network, evaluated in plain C# - no libtorch, no GPU. This is what the desktop app
/// and the CI tests use: a network exported from training (a few hundred kilobytes to a few
/// megabytes) runs anywhere .NET runs.
///
/// Batch normalisation is folded into the preceding convolution at export time
/// (w' = w * gamma / sqrt(var + eps), b' = beta - mean * gamma / sqrt(var + eps)), so inference is
/// just convolutions, ReLUs and fully connected layers.
///
/// Activations are stored pixel-major (HWC: the channels of one cell are contiguous). A 3x3
/// convolution then walks the cells, and for every valid neighbour adds
/// <c>input[neighbour, ci] * W[tap, ci, 0..Cout]</c> into the cell's Cout accumulators: with
/// AVX2 + FMA, 64 output channels are 8 vector registers that stay in registers for the whole
/// 3 x 3 x Cin reduction. Board edges are skipped instead of padded. Positions of a batch run in
/// parallel on all cores.
///
/// File format (.gnet, little endian): "GNET", version, game name, the NetworkSpec integers, then
/// each weight array as (length, floats) in a fixed order - see <see cref="Write"/>.
/// </summary>
public sealed class NativeNetwork
{
    public const int Version = 1;

    public readonly NetworkSpec Spec;
    readonly float[] _stemW, _stemB;
    readonly float[][] _blockW1, _blockB1, _blockW2, _blockB2;
    readonly float[] _polConvW, _polConvB, _polFcW, _polFcB;
    readonly float[] _valConvW, _valConvB, _valFc1W, _valFc1B, _valFc2W, _valFc2B;
    readonly ThreadLocal<Scratch> _scratch;
    readonly int[] _neighbours;   // per cell: 9 neighbour cell indices (-1 outside the board)

    sealed class Scratch
    {
        public readonly float[] Input, A, B, T, Head, Hidden;

        public Scratch(NetworkSpec s)
        {
            int area = s.Area;
            Input = new float[s.Planes * area];
            A = new float[s.Channels * area];
            B = new float[s.Channels * area];
            T = new float[s.Channels * area];
            Head = new float[Math.Max(s.PolicyChannels, s.ValueChannels) * area];
            Hidden = new float[s.ValueHidden];
        }
    }

    public NativeNetwork(NetworkSpec spec, IReadOnlyList<float[]> weights)
    {
        Spec = spec;
        int i = 0;
        float[] Next() => weights[i++];
        int c = spec.Channels;
        _stemW = Transpose3x3(Next(), c, spec.Planes); _stemB = Next();
        _blockW1 = new float[spec.Blocks][]; _blockB1 = new float[spec.Blocks][];
        _blockW2 = new float[spec.Blocks][]; _blockB2 = new float[spec.Blocks][];
        for (int b = 0; b < spec.Blocks; b++)
        {
            _blockW1[b] = Transpose3x3(Next(), c, c); _blockB1[b] = Next();
            _blockW2[b] = Transpose3x3(Next(), c, c); _blockB2[b] = Next();
        }
        _polConvW = Next(); _polConvB = Next(); _polFcW = Next(); _polFcB = Next();
        _valConvW = Next(); _valConvB = Next(); _valFc1W = Next(); _valFc1B = Next(); _valFc2W = Next(); _valFc2B = Next();
        if (i != weights.Count) throw new InvalidDataException($"expected {i} weight arrays, got {weights.Count}");
        int area = spec.Area;
        Check(_polFcW, spec.Actions * spec.PolicyChannels * area);
        Check(_valFc1W, spec.ValueHidden * spec.ValueChannels * area);
        _neighbours = new int[area * 9];
        for (int y = 0; y < spec.Height; y++)
            for (int x = 0; x < spec.Width; x++)
                for (int t = 0; t < 9; t++)
                {
                    int yy = y + t / 3 - 1, xx = x + t % 3 - 1;
                    _neighbours[(y * spec.Width + x) * 9 + t] =
                        yy >= 0 && yy < spec.Height && xx >= 0 && xx < spec.Width ? yy * spec.Width + xx : -1;
                }
        _scratch = new ThreadLocal<Scratch>(() => new Scratch(spec));
    }

    /// <summary>[Cout][Cin][3][3] (PyTorch) -> [tap][Cin][Cout], so a tap's weights for all outputs are contiguous.</summary>
    static float[] Transpose3x3(float[] w, int cout, int cin)
    {
        Check(w, cout * cin * 9);
        var t = new float[w.Length];
        for (int co = 0; co < cout; co++)
            for (int ci = 0; ci < cin; ci++)
                for (int tap = 0; tap < 9; tap++)
                    t[(tap * cin + ci) * cout + co] = w[(co * cin + ci) * 9 + tap];
        return t;
    }

    static void Check(float[] a, int n)
    {
        if (a.Length != n) throw new InvalidDataException($"weight array of {a.Length} values, expected {n}");
    }

    // ---- file format --------------------------------------------------------------------------------
    public static void Write(string path, NetworkSpec spec, IReadOnlyList<float[]> weights)
    {
        using var w = new BinaryWriter(File.Create(path));
        w.Write("GNET"u8);
        w.Write(Version);
        w.Write(spec.Game);
        foreach (int v in new[] { spec.Planes, spec.Height, spec.Width, spec.Actions, spec.Blocks, spec.Channels,
                                  spec.PolicyChannels, spec.ValueChannels, spec.ValueHidden })
            w.Write(v);
        w.Write(weights.Count);
        foreach (var a in weights)
        {
            w.Write(a.Length);
            foreach (float f in a) w.Write(f);
        }
    }

    public static NativeNetwork Load(string path)
    {
        using var stream = File.OpenRead(path);
        return Load(stream, path);
    }

    public static NativeNetwork Load(Stream stream, string name = "network")
    {
        using var r = new BinaryReader(stream);
        if (!r.ReadBytes(4).AsSpan().SequenceEqual("GNET"u8)) throw new InvalidDataException($"{name} is not a .gnet file");
        int version = r.ReadInt32();
        if (version != Version) throw new InvalidDataException($"unsupported .gnet version {version}");
        string game = r.ReadString();
        int[] v = Enumerable.Range(0, 9).Select(_ => r.ReadInt32()).ToArray();
        var spec = new NetworkSpec
        {
            Game = game, Planes = v[0], Height = v[1], Width = v[2], Actions = v[3], Blocks = v[4], Channels = v[5],
            PolicyChannels = v[6], ValueChannels = v[7], ValueHidden = v[8],
        };
        int count = r.ReadInt32();
        var weights = new List<float[]>(count);
        for (int k = 0; k < count; k++)
        {
            var a = new float[r.ReadInt32()];
            var bytes = r.ReadBytes(a.Length * 4);
            Buffer.BlockCopy(bytes, 0, a, 0, bytes.Length);
            weights.Add(a);
        }
        return new NativeNetwork(spec, weights);
    }

    // ---- inference ---------------------------------------------------------------------------------------
    /// <summary>3x3 convolution, zero padding, HWC in and out, + bias, optional ReLU.</summary>
    void Conv3x3(ReadOnlySpan<float> input, int cin, float[] wt, float[] bias, Span<float> output, bool relu)
    {
        int area = Spec.Area, cout = bias.Length;
        for (int p = 0; p < area; p++)
        {
            var o = output.Slice(p * cout, cout);
            bias.CopyTo(o);
            int cb = 0;
            if (Fma.IsSupported && Avx.IsSupported)
                for (; cb + 64 <= cout; cb += 64) Accumulate64(input, cin, wt, cout, p, cb, o.Slice(cb, 64));
            for (; cb < cout; cb++)
            {
                // Generic fallback for channel counts that are not multiples of 64 (or no AVX2).
                float acc = 0;
                for (int t = 0; t < 9; t++)
                {
                    int nb = _neighbours[p * 9 + t];
                    if (nb < 0) continue;
                    for (int ci = 0; ci < cin; ci++) acc += input[nb * cin + ci] * wt[(t * cin + ci) * cout + cb];
                }
                o[cb] += acc;
            }
            if (relu) TensorPrimitives.Max(o, 0f, o);
        }
    }

    /// <summary>64 output channels of one cell: 8 AVX registers accumulate over 9 taps x Cin inputs.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    unsafe void Accumulate64(ReadOnlySpan<float> input, int cin, float[] wt, int cout, int p, int cb, Span<float> o)
    {
        fixed (float* inp = input, w = wt, op = o)
        {
            var a0 = Avx.LoadVector256(op); var a1 = Avx.LoadVector256(op + 8);
            var a2 = Avx.LoadVector256(op + 16); var a3 = Avx.LoadVector256(op + 24);
            var a4 = Avx.LoadVector256(op + 32); var a5 = Avx.LoadVector256(op + 40);
            var a6 = Avx.LoadVector256(op + 48); var a7 = Avx.LoadVector256(op + 56);
            for (int t = 0; t < 9; t++)
            {
                int nb = _neighbours[p * 9 + t];
                if (nb < 0) continue;
                float* x = inp + nb * cin;
                float* wr = w + (long)t * cin * cout + cb;
                for (int ci = 0; ci < cin; ci++, wr += cout)
                {
                    var v = Vector256.Create(x[ci]);
                    a0 = Fma.MultiplyAdd(Avx.LoadVector256(wr), v, a0);
                    a1 = Fma.MultiplyAdd(Avx.LoadVector256(wr + 8), v, a1);
                    a2 = Fma.MultiplyAdd(Avx.LoadVector256(wr + 16), v, a2);
                    a3 = Fma.MultiplyAdd(Avx.LoadVector256(wr + 24), v, a3);
                    a4 = Fma.MultiplyAdd(Avx.LoadVector256(wr + 32), v, a4);
                    a5 = Fma.MultiplyAdd(Avx.LoadVector256(wr + 40), v, a5);
                    a6 = Fma.MultiplyAdd(Avx.LoadVector256(wr + 48), v, a6);
                    a7 = Fma.MultiplyAdd(Avx.LoadVector256(wr + 56), v, a7);
                }
            }
            Avx.Store(op, a0); Avx.Store(op + 8, a1); Avx.Store(op + 16, a2); Avx.Store(op + 24, a3);
            Avx.Store(op + 32, a4); Avx.Store(op + 40, a5); Avx.Store(op + 48, a6); Avx.Store(op + 56, a7);
        }
    }

    /// <summary>1x1 convolution (+ bias, ReLU) from HWC activations to a CHW output (PyTorch's flatten order).</summary>
    void Conv1x1(ReadOnlySpan<float> input, float[] w, float[] b, Span<float> output)
    {
        int area = Spec.Area, cin = Spec.Channels;
        for (int co = 0; co < b.Length; co++)
        {
            var wr = w.AsSpan(co * cin, cin);
            for (int p = 0; p < area; p++)
                output[co * area + p] = MathF.Max(0f, TensorPrimitives.Dot(input.Slice(p * cin, cin), wr) + b[co]);
        }
    }

    /// <summary>Evaluates one position given as CHW input planes; writes the policy logits and returns the value.</summary>
    public float Evaluate(ReadOnlySpan<float> planes, Span<float> logits)
    {
        var s = _scratch.Value!;
        var spec = Spec;
        int area = spec.Area, c = spec.Channels, pl = spec.Planes;
        // CHW -> HWC
        for (int k = 0; k < pl; k++)
            for (int p = 0; p < area; p++)
                s.Input[p * pl + k] = planes[k * area + p];
        Conv3x3(s.Input, pl, _stemW, _stemB, s.A, true);
        for (int b = 0; b < spec.Blocks; b++)
        {
            Conv3x3(s.A, c, _blockW1[b], _blockB1[b], s.T, true);
            Conv3x3(s.T, c, _blockW2[b], _blockB2[b], s.B, false);
            var a = s.A.AsSpan(0, c * area);
            TensorPrimitives.Add(a, s.B.AsSpan(0, c * area), a);
            TensorPrimitives.Max(a, 0f, a);
        }
        // policy head
        var head = s.Head.AsSpan(0, spec.PolicyChannels * area);
        Conv1x1(s.A, _polConvW, _polConvB, head);
        int pin = head.Length;
        for (int o = 0; o < spec.Actions; o++)
            logits[o] = TensorPrimitives.Dot(_polFcW.AsSpan(o * pin, pin), head) + _polFcB[o];
        // value head
        head = s.Head.AsSpan(0, spec.ValueChannels * area);
        Conv1x1(s.A, _valConvW, _valConvB, head);
        int vin = head.Length;
        for (int o = 0; o < spec.ValueHidden; o++)
            s.Hidden[o] = MathF.Max(0f, TensorPrimitives.Dot(_valFc1W.AsSpan(o * vin, vin), head) + _valFc1B[o]);
        return MathF.Tanh(TensorPrimitives.Dot(_valFc2W, s.Hidden) + _valFc2B[0]);
    }

    /// <summary>An <see cref="IEvaluator{TGame}"/> backed by this network.</summary>
    public IEvaluator<TGame> For<TGame>() where TGame : struct, IGame<TGame>
    {
        if (Spec.Game != TGame.Name || Spec.Actions != TGame.ActionCount || Spec.Planes != TGame.PlaneCount)
            throw new InvalidOperationException($"this network was trained for {Spec.Game}, not {TGame.Name}");
        return new Evaluator<TGame>(this);
    }

    sealed class Evaluator<TGame>(NativeNetwork net) : IEvaluator<TGame> where TGame : struct, IGame<TGame>
    {
        readonly ThreadLocal<float[]> _planes = new(() => new float[TGame.PlaneCount * TGame.Height * TGame.Width]);

        public void Evaluate(ReadOnlySpan<TGame> states, Span<float> logits, Span<float> values)
        {
            int a = TGame.ActionCount;
            if (states.Length == 1)
            {
                states[0].Encode(_planes.Value!);
                values[0] = net.Evaluate(_planes.Value!, logits[..a]);
                return;
            }
            var st = states.ToArray();
            var lg = new float[st.Length * a];
            var vs = new float[st.Length];
            Parallel.For(0, st.Length, i =>
            {
                var p = _planes.Value!;
                st[i].Encode(p);
                vs[i] = net.Evaluate(p, lg.AsSpan(i * a, a));
            });
            lg.CopyTo(logits);
            vs.CopyTo(values);
        }
    }
}
