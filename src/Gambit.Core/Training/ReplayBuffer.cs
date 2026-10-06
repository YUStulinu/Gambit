using System.Runtime.InteropServices;
using Gambit.Core.Games;

namespace Gambit.Core.Training;

/// <summary>
/// The training window: the most recent self-play positions, oldest dropped first.
/// Batches are drawn uniformly at random, each position under a random board symmetry
/// (the game's rules are symmetric, so a mirrored position with a mirrored policy is an equally
/// valid example - free extra data).
/// </summary>
public sealed class ReplayBuffer<TGame> where TGame : unmanaged, IGame<TGame>
{
    readonly List<Sample<TGame>> _samples = [];
    readonly int _capacity;
    int _start;   // ring-buffer head once full

    public ReplayBuffer(int capacity) => _capacity = capacity;

    public int Count => _samples.Count;
    public int Capacity => _capacity;

    public void Add(Sample<TGame> s)
    {
        if (_samples.Count < _capacity) _samples.Add(s);
        else
        {
            _samples[_start] = s;
            _start = (_start + 1) % _capacity;
        }
    }

    public void AddGame(GameRecord<TGame> g)
    {
        foreach (var s in g.Samples) Add(s);
    }

    public Sample<TGame> this[int i] => _samples[i];

    /// <summary>
    /// Fills one training batch: input planes (batch x planes x H x W), policy targets (batch x actions)
    /// and value targets, each sample under a random symmetry.
    /// </summary>
    public void FillBatch(Random rng, int batch, float[] planes, float[] policies, float[] values)
    {
        int planeSize = TGame.PlaneCount * TGame.Height * TGame.Width, a = TGame.ActionCount;
        var tmp = new float[planeSize];
        for (int i = 0; i < batch; i++)
        {
            var s = _samples[rng.Next(_samples.Count)];
            int sym = rng.Next(TGame.SymmetryCount);
            Symmetry.Encode(s.State, sym, planes.AsSpan(i * planeSize, planeSize), tmp);
            var pol = policies.AsSpan(i * a, a);
            for (int k = 0; k < a; k++) pol[TGame.TransformAction(k, sym)] = s.Policy[k];
            values[i] = s.ValueTarget;
        }
    }

    // ---- persistence: one file per generation ----------------------------------------------------------
    public static void Save(string path, IEnumerable<GameRecord<TGame>> games)
    {
        using var w = new BinaryWriter(File.Create(path));
        w.Write(TGame.Name);
        var list = games.ToList();
        w.Write(list.Count);
        foreach (var g in list)
        {
            w.Write(g.Result);
            w.Write(g.Moves.Count);
            foreach (int m in g.Moves) w.Write((short)m);
            w.Write(g.Samples.Count);
            foreach (var s in g.Samples)
            {
                var st = s.State;
                w.Write(MemoryMarshal.AsBytes(new ReadOnlySpan<TGame>(ref st)));
                w.Write(s.RootValue);
                w.Write(s.ValueTarget);
                foreach (float p in s.Policy) w.Write(p);
            }
        }
    }

    public static List<GameRecord<TGame>> Load(string path)
    {
        using var r = new BinaryReader(File.OpenRead(path));
        if (r.ReadString() != TGame.Name) throw new InvalidDataException($"{path} holds games of another type");
        int n = r.ReadInt32();
        var games = new List<GameRecord<TGame>>(n);
        int size = Marshal.SizeOf<TGame>();
        for (int i = 0; i < n; i++)
        {
            var g = new GameRecord<TGame> { Result = r.ReadInt32() };
            int moves = r.ReadInt32();
            for (int k = 0; k < moves; k++) g.Moves.Add(r.ReadInt16());
            int samples = r.ReadInt32();
            for (int k = 0; k < samples; k++)
            {
                var st = MemoryMarshal.Read<TGame>(r.ReadBytes(size));
                float rootValue = r.ReadSingle(), target = r.ReadSingle();
                var policy = new float[TGame.ActionCount];
                for (int j = 0; j < policy.Length; j++) policy[j] = r.ReadSingle();
                g.Samples.Add(new Sample<TGame>(st, policy, rootValue) { ValueTarget = target });
            }
            games.Add(g);
        }
        return games;
    }
}

public static class Symmetry
{
    /// <summary>Encodes <paramref name="state"/> and maps every cell through symmetry <paramref name="sym"/>.</summary>
    public static void Encode<TGame>(in TGame state, int sym, Span<float> planes, Span<float> tmp)
        where TGame : struct, IGame<TGame>
    {
        if (sym == 0)
        {
            state.Encode(planes);
            return;
        }
        state.Encode(tmp);
        int area = TGame.Height * TGame.Width;
        for (int p = 0; p < TGame.PlaneCount; p++)
            for (int c = 0; c < area; c++)
                planes[p * area + TGame.TransformCell(c, sym)] = tmp[p * area + c];
    }
}
