using System.Globalization;
using System.Text;

namespace Gambit.Cli;

/// <summary>Self-contained SVG line charts of a training run (no plotting library needed).</summary>
public static class Charts
{
    public sealed record Series(string Label, string Color, IReadOnlyList<(double X, double Y)> Points, bool Dashed = false);

    static string F(double v) => v.ToString("0.##", CultureInfo.InvariantCulture);

    /// <summary>Draws one panel at (x0, y0) of size (w, h).</summary>
    static void Panel(StringBuilder sb, double x0, double y0, double w, double h, string title, string xLabel,
                      IReadOnlyList<Series> series, double? yMin = null, double? yMax = null, string yFormat = "0", double yScale = 1)
    {
        var all = series.SelectMany(s => s.Points).Where(p => double.IsFinite(p.Y)).ToList();
        if (all.Count == 0) return;
        double minX = all.Min(p => p.X), maxX = Math.Max(all.Max(p => p.X), minX + 1);
        double minY = yMin ?? all.Min(p => p.Y), maxY = yMax ?? all.Max(p => p.Y);
        if (maxY - minY < 1e-9) { minY -= 1; maxY += 1; }
        double pad = (maxY - minY) * 0.06;
        if (yMin == null) minY -= pad;
        if (yMax == null) maxY += pad;
        const double left = 58, right = 14, top = 30, bottom = 40;
        double pw = w - left - right, ph = h - top - bottom;
        double Sx(double x) => x0 + left + (x - minX) / (maxX - minX) * pw;
        double Sy(double y) => y0 + top + (1 - (y - minY) / (maxY - minY)) * ph;

        sb.Append($"<text x='{F(x0 + left)}' y='{F(y0 + 18)}' class='title'>{title}</text>");
        for (int i = 0; i <= 4; i++)
        {
            double y = minY + (maxY - minY) * i / 4;
            sb.Append($"<line x1='{F(x0 + left)}' x2='{F(x0 + left + pw)}' y1='{F(Sy(y))}' y2='{F(Sy(y))}' class='grid'/>");
            sb.Append($"<text x='{F(x0 + left - 6)}' y='{F(Sy(y) + 4)}' class='tick' text-anchor='end'>{(y * yScale).ToString(yFormat, CultureInfo.InvariantCulture)}</text>");
        }
        // Integer ticks at a round step (1, 2, 5, 10, 20, ...).
        double span = maxX - minX, step = 1;
        foreach (double c in new double[] { 1, 2, 5, 10, 20, 25, 50, 100, 200, 500 })
            if (span / c <= 8) { step = c; break; }
        for (double x = Math.Ceiling(minX / step) * step; x <= maxX + 1e-9; x += step)
            sb.Append($"<text x='{F(Sx(x))}' y='{F(y0 + top + ph + 16)}' class='tick' text-anchor='middle'>{x:0}</text>");
        sb.Append($"<text x='{F(x0 + left + pw / 2)}' y='{F(y0 + h - 6)}' class='tick' text-anchor='middle'>{xLabel}</text>");
        sb.Append($"<rect x='{F(x0 + left)}' y='{F(y0 + top)}' width='{F(pw)}' height='{F(ph)}' class='frame'/>");
        double ly = y0 + top + 14;
        foreach (var s in series)
        {
            var pts = s.Points.Where(p => double.IsFinite(p.Y)).ToList();
            if (pts.Count == 0) continue;
            string path = string.Join(" ", pts.Select((p, i) => $"{(i == 0 ? "M" : "L")}{F(Sx(p.X))},{F(Sy(p.Y))}"));
            sb.Append($"<path d='{path}' fill='none' stroke='{s.Color}' stroke-width='2'{(s.Dashed ? " stroke-dasharray='5,4'" : "")}/>");
            if (pts.Count < 40)
                foreach (var p in pts) sb.Append($"<circle cx='{F(Sx(p.X))}' cy='{F(Sy(p.Y))}' r='2.6' fill='{s.Color}'/>");
            if (series.Count > 1)
            {
                sb.Append($"<line x1='{F(x0 + left + 10)}' x2='{F(x0 + left + 28)}' y1='{F(ly - 4)}' y2='{F(ly - 4)}' stroke='{s.Color}' stroke-width='2'{(s.Dashed ? " stroke-dasharray='5,4'" : "")}/>");
                sb.Append($"<text x='{F(x0 + left + 34)}' y='{F(ly)}' class='legend'>{s.Label}</text>");
                ly += 16;
            }
        }
    }

    public static void WriteProgress(string path, string game, IReadOnlyList<TrainCommand.GenerationLog> logs,
                                     IReadOnlyDictionary<string, double> ratings,
                                     IReadOnlyDictionary<string, double>? tournament = null)
    {
        const double W = 1100, H = 720;
        var sb = new StringBuilder();
        sb.Append($"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' font-family='Segoe UI, Helvetica, Arial, sans-serif'>");
        sb.Append("<style>.title{font-size:14px;font-weight:600;fill:#222}.tick{font-size:11px;fill:#555}.legend{font-size:11px;fill:#333}" +
                  ".grid{stroke:#e5e5e5}.frame{fill:none;stroke:#bbb}</style><rect width='100%' height='100%' fill='white'/>");

        // Elo of every accepted network, from the global table.
        var elo = ratings.Where(kv => kv.Key.StartsWith("gen-"))
                         .Select(kv => (X: (double)int.Parse(kv.Key[4..]), Y: kv.Value)).OrderBy(p => p.X).ToList();
        var anchorLines = TrainCommand.Anchors
            .Where(a => ratings.ContainsKey(a.Name))
            .Select((a, i) => new Series($"{a.Name} = {ratings[a.Name]:F0}", i == 0 ? "#999" : "#555",
                                         [(0, ratings[a.Name]), (Math.Max(1, logs.Count > 0 ? logs[^1].Generation : 1), ratings[a.Name])], true))
            .ToList();
        var series = new List<Series> { new("gating chain", "#d9480f", elo) };
        if (tournament != null)
            series.Add(new Series("round-robin tournament", "#1971c2",
                tournament.Where(kv => kv.Key.StartsWith("gen-")).Select(kv => ((double)int.Parse(kv.Key[4..]), kv.Value))
                          .OrderBy(p => p.Item1).ToList()));
        Panel(sb, 0, 0, W / 2, H / 2, $"{game}: Elo of the accepted networks", "generation", [.. series, .. anchorLines]);

        var acc = logs.Where(l => l.SolverOptimal.HasValue).ToList();
        if (acc.Count > 0)
            Panel(sb, W / 2, 0, W / 2, H / 2, "moves graded by the perfect solver (1000 positions)", "generation",
                  [new Series("optimal move, MCTS", "#1971c2", acc.Select(l => ((double)l.Generation, l.SolverOptimal!.Value)).ToList()),
                   new Series("optimal move, raw network", "#74c0fc", acc.Select(l => ((double)l.Generation, l.SolverOptimalRaw ?? double.NaN)).ToList()),
                   new Series("keeps the theoretical result, MCTS", "#2b8a3e", acc.Select(l => ((double)l.Generation, l.SolverKeepsResult ?? double.NaN)).ToList())],
                  null, 1, "0", 100);
        else
            Panel(sb, W / 2, 0, W / 2, H / 2, "arena score of the candidate", "generation",
                  [new Series("score vs best", "#1971c2", logs.Select(l => ((double)l.Generation, l.ArenaScore)).ToList())], 0, 1, "0", 100);

        Panel(sb, 0, H / 2, W / 2, H / 2, "training loss", "generation",
              [new Series("policy (cross-entropy)", "#7048e8", logs.Select(l => ((double)l.Generation, l.PolicyLoss)).ToList()),
               new Series("value (MSE)", "#e8590c", logs.Select(l => ((double)l.Generation, l.ValueLoss)).ToList()),
               new Series("policy target entropy", "#adb5bd", logs.Select(l => ((double)l.Generation, l.TargetEntropy)).ToList(), true)],
              0, null, "0.00");
        Panel(sb, W / 2, H / 2, W / 2, H / 2, "self-play games", "generation",
              [new Series("first player wins", "#2f9e44", logs.Select(l => ((double)l.Generation, l.FirstWinRate)).ToList()),
               new Series("draws", "#868e96", logs.Select(l => ((double)l.Generation, l.DrawRate)).ToList())],
              0, 1, "0", 100);
        sb.Append("</svg>");
        File.WriteAllText(path, sb.ToString());
    }
}
