using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Gambit.Core.Session;

namespace Gambit.App;

/// <summary>
/// Draws the board and, optionally, what the AI is considering: for every candidate move the share
/// of the search's visits (bigger marker = more attention) and its estimated win chance.
/// </summary>
public sealed class BoardControl : Control
{
    IGameSession? _session;
    Analysis? _analysis;
    (int X, int Y)? _hover;

    public bool AcceptsInput { get; set; } = true;

    /// <summary>Raised with the action of a click on a legal move.</summary>
    public event Action<int>? MoveClicked;

    public IGameSession? Session
    {
        get => _session;
        set
        {
            _session = value;
            _analysis = null;
            InvalidateVisual();
        }
    }

    public void ShowAnalysis(Analysis? analysis)
    {
        _analysis = analysis;
        InvalidateVisual();
    }

    public void Refresh() => InvalidateVisual();

    static readonly IBrush C4Board = new SolidColorBrush(Color.Parse("#1c5fd6"));
    static readonly IBrush C4Hole = new SolidColorBrush(Color.Parse("#eef2f8"));
    static readonly IBrush Red = new RadialGradientBrush
    {
        GradientStops = { new GradientStop(Color.Parse("#ff6b6b"), 0), new GradientStop(Color.Parse("#d42a2a"), 1) },
    };
    static readonly IBrush Yellow = new RadialGradientBrush
    {
        GradientStops = { new GradientStop(Color.Parse("#ffe066"), 0), new GradientStop(Color.Parse("#f2b705"), 1) },
    };
    static readonly IBrush Wood = new LinearGradientBrush
    {
        StartPoint = new RelativePoint(0, 0, RelativeUnit.Relative),
        EndPoint = new RelativePoint(1, 1, RelativeUnit.Relative),
        GradientStops = { new GradientStop(Color.Parse("#e8c27a"), 0), new GradientStop(Color.Parse("#d4a752"), 1) },
    };
    static readonly IBrush Black = new RadialGradientBrush
    {
        GradientOrigin = new RelativePoint(0.35, 0.3, RelativeUnit.Relative),
        GradientStops = { new GradientStop(Color.Parse("#666"), 0), new GradientStop(Color.Parse("#111"), 1) },
    };
    static readonly IBrush White = new RadialGradientBrush
    {
        GradientOrigin = new RelativePoint(0.35, 0.3, RelativeUnit.Relative),
        GradientStops = { new GradientStop(Colors.White, 0), new GradientStop(Color.Parse("#cfcfcf"), 1) },
    };
    static readonly IPen GridPen = new Pen(new SolidColorBrush(Color.Parse("#5c4520")), 1.2);
    static readonly IPen LastMovePen = new Pen(new SolidColorBrush(Color.Parse("#20c997")), 3);
    static readonly IPen WinPen = new Pen(new SolidColorBrush(Color.Parse("#ffffff")), 4);
    static readonly IBrush Heat = new SolidColorBrush(Color.Parse("#e8590c"));

    (double Cell, double OffsetX, double OffsetY, double Header) Layout()
    {
        var s = _session!;
        double header = s.ColumnGame ? 1.0 : 0;   // a strip above Connect Four's board for the column statistics
        double cell = Math.Min(Bounds.Width / s.Width, Bounds.Height / (s.Height + header));
        double ox = (Bounds.Width - cell * s.Width) / 2;
        double oy = (Bounds.Height - cell * (s.Height + header)) / 2 + header * cell;
        return (cell, ox, oy, header * cell);
    }

    (int X, int Y)? CellAtPoint(Point p)
    {
        if (_session == null) return null;
        var (cell, ox, oy, _) = Layout();
        int x = (int)Math.Floor((p.X - ox) / cell), y = (int)Math.Floor((p.Y - oy) / cell);
        if (_session.ColumnGame) y = Math.Clamp(y, 0, _session.Height - 1);
        if (x < 0 || x >= _session.Width || y < 0 || y >= _session.Height) return null;
        return (x, y);
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        var c = CellAtPoint(e.GetPosition(this));
        if (c != _hover)
        {
            _hover = c;
            InvalidateVisual();
        }
    }

    protected override void OnPointerExited(PointerEventArgs e)
    {
        base.OnPointerExited(e);
        _hover = null;
        InvalidateVisual();
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        if (!AcceptsInput || _session == null || _session.IsOver) return;
        if (CellAtPoint(e.GetPosition(this)) is not { } c) return;
        int action = _session.ActionAt(c.X, c.Y);
        if (action >= 0) MoveClicked?.Invoke(action);
    }

    public override void Render(DrawingContext ctx)
    {
        if (_session == null) return;
        if (_session.ColumnGame) RenderConnectFour(ctx);
        else RenderGomoku(ctx);
    }

    void RenderConnectFour(DrawingContext ctx)
    {
        var s = _session!;
        var (cell, ox, oy, header) = Layout();
        var board = new Rect(ox, oy, cell * s.Width, cell * s.Height);
        ctx.DrawRectangle(C4Board, null, board, cell * 0.15, cell * 0.15);
        double r = cell * 0.40;

        // Hover: a translucent stone where the move would land.
        int hoverAction = AcceptsInput && !s.IsOver && _hover is { } h ? s.ActionAt(h.X, h.Y) : -1;
        var win = s.WinningLine;
        var last = s.LastCell;
        for (int y = 0; y < s.Height; y++)
            for (int x = 0; x < s.Width; x++)
            {
                var center = new Point(ox + (x + 0.5) * cell, oy + (y + 0.5) * cell);
                int stone = s.CellAt(x, y);
                IBrush fill = stone == 1 ? Red : stone == 2 ? Yellow : C4Hole;
                ctx.DrawEllipse(fill, null, center, r, r);
                if (last is { } l && l.X == x && l.Y == y) ctx.DrawEllipse(null, LastMovePen, center, r * 0.55, r * 0.55);
                if (win.Contains((x, y))) ctx.DrawEllipse(null, WinPen, center, r * 0.75, r * 0.75);
            }
        if (hoverAction >= 0)
        {
            var (hx, hy) = s.CellOf(hoverAction);
            var ghost = s.SideToMove == 0 ? Red : Yellow;
            using (ctx.PushOpacity(0.45))
                ctx.DrawEllipse(ghost, null, new Point(ox + (hx + 0.5) * cell, oy + (hy + 0.5) * cell), r, r);
        }

        // Column statistics above the board: visit share (bar) and win chance.
        if (_analysis is { } a && a.TotalVisits > 0)
        {
            double total = a.TotalVisits;
            int best = a.Moves.MaxBy(m => m.Visits).Action;
            foreach (var m in a.Moves)
            {
                double share = m.Visits / total;
                double x0 = ox + m.Action * cell;
                double barH = (header - 22) * share;
                var bar = new Rect(x0 + cell * 0.2, oy - 20 - barH, cell * 0.6, Math.Max(barH, 1));
                ctx.DrawRectangle(m.Action == best ? Heat : new SolidColorBrush(Color.Parse("#ffa94d")), null, bar, 3, 3);
                string label = $"{100 * share:F0}%";
                DrawText(ctx, label, new Point(x0 + cell / 2, oy - 20 - barH - 9), 12, Brushes.Black, true);
                if (!float.IsNaN(m.Q))
                    DrawText(ctx, $"{50 * (m.Q + 1):F0}", new Point(x0 + cell / 2, oy - 9), 11, new SolidColorBrush(Color.Parse("#495057")), false);
            }
        }
    }

    void RenderGomoku(DrawingContext ctx)
    {
        var s = _session!;
        var (cell, ox, oy, _) = Layout();
        var board = new Rect(ox, oy, cell * s.Width, cell * s.Height);
        ctx.DrawRectangle(Wood, null, board, 6, 6);
        for (int i = 0; i < s.Width; i++)
        {
            double p = ox + (i + 0.5) * cell;
            ctx.DrawLine(GridPen, new Point(p, oy + cell / 2), new Point(p, oy + cell * (s.Height - 0.5)));
        }
        for (int i = 0; i < s.Height; i++)
        {
            double p = oy + (i + 0.5) * cell;
            ctx.DrawLine(GridPen, new Point(ox + cell / 2, p), new Point(ox + cell * (s.Width - 0.5), p));
        }
        var dark = new SolidColorBrush(Color.Parse("#5c4520"));
        foreach (var (sx, sy) in new[] { (2, 2), (6, 2), (4, 4), (2, 6), (6, 6) })
            ctx.DrawEllipse(dark, null, new Point(ox + (sx + 0.5) * cell, oy + (sy + 0.5) * cell), 3.5, 3.5);
        for (int i = 0; i < s.Width; i++)
        {
            DrawText(ctx, ((char)('a' + i)).ToString(), new Point(ox + (i + 0.5) * cell, oy + cell * s.Height + 10), 11, dark, false);
            DrawText(ctx, (s.Height - i).ToString(), new Point(ox - 10, oy + (i + 0.5) * cell), 11, dark, false);
        }

        double r = cell * 0.44;
        var win = s.WinningLine;
        var last = s.LastCell;
        for (int y = 0; y < s.Height; y++)
            for (int x = 0; x < s.Width; x++)
            {
                int stone = s.CellAt(x, y);
                if (stone == 0) continue;
                var center = new Point(ox + (x + 0.5) * cell, oy + (y + 0.5) * cell);
                ctx.DrawEllipse(stone == 1 ? Black : White, new Pen(Brushes.Black, 0.6), center, r, r);
                if (last is { } l && l.X == x && l.Y == y)
                    ctx.DrawEllipse(stone == 1 ? Brushes.White : Brushes.Black, null, center, r * 0.18, r * 0.18);
                if (win.Contains((x, y))) ctx.DrawEllipse(null, new Pen(Heat, 3), center, r * 0.8, r * 0.8);
            }

        if (_analysis is { } a && a.TotalVisits > 0)
        {
            double total = a.TotalVisits;
            int best = a.Moves.MaxBy(m => m.Visits).Action;
            foreach (var m in a.Moves.OrderByDescending(m => m.Visits).Take(12))
            {
                double share = m.Visits / total;
                if (share < 0.01) continue;
                var (x, y) = s.CellOf(m.Action);
                var center = new Point(ox + (x + 0.5) * cell, oy + (y + 0.5) * cell);
                double rr = r * (0.35 + 0.65 * Math.Sqrt(share));
                using (ctx.PushOpacity(m.Action == best ? 0.85 : 0.55))
                    ctx.DrawEllipse(Heat, null, center, rr, rr);
                DrawText(ctx, $"{100 * share:F0}", center, 11, Brushes.White, true);
            }
        }
        else if (AcceptsInput && !s.IsOver && _hover is { } h && s.ActionAt(h.X, h.Y) >= 0)
        {
            using (ctx.PushOpacity(0.45))
                ctx.DrawEllipse(s.SideToMove == 0 ? Black : White, null,
                                new Point(ox + (h.X + 0.5) * cell, oy + (h.Y + 0.5) * cell), r, r);
        }
    }

    static void DrawText(DrawingContext ctx, string text, Point center, double size, IBrush brush, bool bold)
    {
        var ft = new FormattedText(text, System.Globalization.CultureInfo.InvariantCulture, FlowDirection.LeftToRight,
                                   new Typeface("Inter", FontStyle.Normal, bold ? FontWeight.SemiBold : FontWeight.Normal), size, brush);
        ctx.DrawText(ft, new Point(center.X - ft.Width / 2, center.Y - ft.Height / 2));
    }
}
