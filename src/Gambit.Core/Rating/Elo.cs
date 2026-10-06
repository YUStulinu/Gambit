namespace Gambit.Core.Rating;

/// <summary>
/// Elo ratings from a set of match results, by maximum likelihood (the Bradley-Terry model):
/// player i beats player j with probability 1 / (1 + 10^((Rj - Ri) / 400)), a draw counting as
/// half a win for each. The ratings are those that make all the observed results most likely,
/// computed with Hunter's MM algorithm, then shifted so that <c>anchor</c> has rating
/// <c>anchorRating</c>. Every pair that played also gets one virtual draw, a weak prior that keeps
/// a player that has won every game at a finite rating.
/// </summary>
public sealed class EloTable
{
    readonly Dictionary<(string, string), (double Score, int Games)> _results = new();
    readonly HashSet<string> _players = [];

    public IReadOnlyCollection<string> Players => _players;

    /// <summary>Records games between <paramref name="a"/> and <paramref name="b"/>.</summary>
    public void Add(string a, string b, int winsA, int draws, int winsB)
    {
        if (a == b) return;
        _players.Add(a);
        _players.Add(b);
        Accumulate(a, b, winsA + 0.5 * draws, winsA + draws + winsB);
        Accumulate(b, a, winsB + 0.5 * draws, winsA + draws + winsB);
    }

    void Accumulate(string a, string b, double score, int games)
    {
        var cur = _results.GetValueOrDefault((a, b));
        _results[(a, b)] = (cur.Score + score, cur.Games + games);
    }

    public (double Score, int Games) Between(string a, string b) => _results.GetValueOrDefault((a, b));

    public Dictionary<string, double> Compute(string anchor, double anchorRating = 0, int iterations = 2000)
    {
        var names = _players.ToList();
        int n = names.Count;
        var index = names.Select((p, i) => (p, i)).ToDictionary(t => t.p, t => t.i);
        var score = new double[n];          // total score, including the prior
        var games = new double[n, n];
        foreach (var ((a, b), (s, g)) in _results)
        {
            int i = index[a], j = index[b];
            score[i] += s + 0.5;            // + one virtual draw per pair
            games[i, j] += g + 1;
        }
        var gamma = Enumerable.Repeat(1.0, n).ToArray();
        for (int it = 0; it < iterations; it++)
        {
            double maxChange = 0;
            for (int i = 0; i < n; i++)
            {
                double denom = 0;
                for (int j = 0; j < n; j++)
                    if (games[i, j] > 0) denom += games[i, j] / (gamma[i] + gamma[j]);
                if (denom <= 0) continue;
                double next = score[i] / denom;
                maxChange = Math.Max(maxChange, Math.Abs(Math.Log(next / gamma[i])));
                gamma[i] = next;
            }
            // Normalise to keep the numbers in range (only ratios matter).
            double mean = Math.Exp(gamma.Average(Math.Log));
            for (int i = 0; i < n; i++) gamma[i] /= mean;
            if (maxChange < 1e-10) break;
        }
        double offset = index.TryGetValue(anchor, out int k) ? anchorRating - 400 * Math.Log10(gamma[k]) : 0;
        return names.ToDictionary(p => p, p => 400 * Math.Log10(gamma[index[p]]) + offset);
    }

    /// <summary>Expected score of a player rated <paramref name="ra"/> against one rated <paramref name="rb"/>.</summary>
    public static double Expected(double ra, double rb) => 1 / (1 + Math.Pow(10, (rb - ra) / 400));

    /// <summary>The Elo difference implied by a score fraction (e.g. 0.75 -> +191).</summary>
    public static double DifferenceFromScore(double fraction)
    {
        fraction = Math.Clamp(fraction, 1e-3, 1 - 1e-3);
        return -400 * Math.Log10(1 / fraction - 1);
    }

    // ---- persistence -------------------------------------------------------------------------------------
    public void Save(string path) =>
        File.WriteAllLines(path, _results.Where(kv => string.CompareOrdinal(kv.Key.Item1, kv.Key.Item2) < 0)
            .Select(kv =>
            {
                var (a, b) = kv.Key;
                var (s, g) = kv.Value;
                return $"{a}\t{b}\t{s.ToString(System.Globalization.CultureInfo.InvariantCulture)}\t{g}";
            }));

    public static EloTable Load(string path)
    {
        var t = new EloTable();
        if (!File.Exists(path)) return t;
        foreach (var line in File.ReadLines(path))
        {
            var f = line.Split('\t');
            if (f.Length != 4) continue;
            double s = double.Parse(f[2], System.Globalization.CultureInfo.InvariantCulture);
            int g = int.Parse(f[3]);
            t._players.Add(f[0]);
            t._players.Add(f[1]);
            t.Accumulate(f[0], f[1], s, g);
            t.Accumulate(f[1], f[0], g - s, g);
        }
        return t;
    }
}
