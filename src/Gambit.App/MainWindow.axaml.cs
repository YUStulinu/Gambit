using Avalonia.Controls;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Threading;
using Gambit.Core.Session;

namespace Gambit.App;

public partial class MainWindow : Window
{
    static readonly int[] Strengths = [1, 50, 400, 1600, 6400];
    static readonly string[] GameKeys = ["connect4", "gomoku"];

    IGameSession _session;
    CancellationTokenSource? _thinking;
    bool _busy;
    int _searchId;    // progress updates of a finished search are ignored
    bool _suppress;   // ignore selection events while the startup options are applied

    public MainWindow()
    {
        InitializeComponent();
        _session = CreateSession(0);
        Board.Session = _session;
        Board.MoveClicked += OnHumanMove;
        GameBox.SelectionChanged += (_, _) => { if (!_suppress) SwitchGame(); };
        SideBox.SelectionChanged += (_, _) => { if (!_suppress) NewGame(); };
        NewGameButton.Click += (_, _) => NewGame();
        UndoButton.Click += (_, _) => Undo();
        HintButton.Click += async (_, _) => await Hint();
        StopButton.Click += (_, _) => _thinking?.Cancel();
        ShowThinking.IsCheckedChanged += (_, _) => { if (ShowThinking.IsChecked != true) Board.ShowAnalysis(null); };
        Opened += async (_, _) => await Startup();
    }

    /// <summary>Applies the command-line options (see Program.Options), otherwise starts a game.</summary>
    async Task Startup()
    {
        var o = Program.Options;
        _suppress = true;
        if (o.TryGetValue("game", out var game))
        {
            GameBox.SelectedIndex = Math.Max(0, Array.IndexOf(GameKeys, game));
            _session = CreateSession(GameBox.SelectedIndex);
            Board.Session = _session;
        }
        if (o.TryGetValue("side", out var side)) SideBox.SelectedIndex = int.Parse(side);
        if (o.TryGetValue("strength", out var st)) StrengthBox.SelectedIndex = int.Parse(st);
        _suppress = false;
        if (!o.TryGetValue("moves", out var moves))
        {
            NewGame();
        }
        else
        {
            _session.Reset();
            NetworkText.Text = $"AI: {_session.NetworkDescription}";
            var tokens = moves.Contains(',') ? moves.Split(',') : moves.Select(c => c.ToString()).ToArray();
            foreach (var t in tokens)
            {
                int a = Enumerable.Range(0, _session.Width * _session.Height).FirstOrDefault(x => _session.IsLegal(x) && _session.ActionName(x) == t, -1);
                if (a < 0) break;
                _session.Play(a);
            }
            UpdateStatus();
        }
        if (o.ContainsKey("hint")) await Hint();
        if (o.TryGetValue("screenshot", out var path))
        {
            await Task.Delay(400);
            var size = new Avalonia.PixelSize((int)Bounds.Width, (int)Bounds.Height);
            using var bmp = new Avalonia.Media.Imaging.RenderTargetBitmap(size);
            bmp.Render(this);
            bmp.Save(path);
            Close();
        }
    }

    static string? FindModel(string game)
    {
        foreach (var dir in new[] { Path.Combine(AppContext.BaseDirectory, "models"), "models", Path.Combine("..", "..", "models") })
        {
            string p = Path.Combine(dir, $"{game}.gnet");
            if (File.Exists(p)) return p;
        }
        return null;
    }

    static IGameSession CreateSession(int index) => Sessions.Create(GameKeys[index], FindModel(GameKeys[index]));

    int HumanSide => SideBox.SelectedIndex;    // 0 first, 1 second, 2 both
    int Simulations => Strengths[Math.Clamp(StrengthBox.SelectedIndex, 0, Strengths.Length - 1)];
    bool AiToMove => !_session.IsOver && HumanSide != 2 && _session.SideToMove != HumanSide;

    string PlayerName(int side) => _session.ColumnGame ? (side == 0 ? "Red" : "Yellow") : (side == 0 ? "Black" : "White");

    void SwitchGame()
    {
        _thinking?.Cancel();
        _session = CreateSession(GameBox.SelectedIndex);
        Board.Session = _session;
        NewGame();
    }

    void NewGame()
    {
        if (_busy)
        {
            _thinking?.Cancel();
            Dispatcher.UIThread.Post(NewGame, DispatcherPriority.Background);
            return;
        }
        _session.Reset();
        Board.ShowAnalysis(null);
        ClearAnalysis();
        NetworkText.Text = $"AI: {_session.NetworkDescription}";
        UpdateStatus();
        _ = MaybeAiMove();
    }

    void Undo()
    {
        if (_busy) return;
        // Take back the AI's reply too, so it is the human's turn again.
        _session.Undo();
        if (AiToMove && _session.Moves.Count > 0) _session.Undo();
        Board.ShowAnalysis(null);
        ClearAnalysis();
        UpdateStatus();
        _ = MaybeAiMove();
    }

    async void OnHumanMove(int action)
    {
        if (_busy || AiToMove) return;
        _session.Play(action);
        Board.ShowAnalysis(null);
        UpdateStatus();
        await MaybeAiMove();
    }

    async Task MaybeAiMove()
    {
        if (!AiToMove || _busy) return;
        var analysis = await Think(Simulations);
        if (analysis == null || _session.IsOver) return;
        int best = analysis.Moves.MaxBy(m => m.Visits).Action;
        _session.Play(best);
        Board.ShowAnalysis(null);
        ShowAnalysisPanel(analysis, forPlayedMove: best);
        UpdateStatus();
    }

    async Task Hint()
    {
        if (_busy || _session.IsOver) return;
        var analysis = await Think(Math.Max(Simulations, 400), hint: true);
        if (analysis != null) Board.ShowAnalysis(analysis);
    }

    /// <summary>Runs the search in the background, showing its progress live.</summary>
    async Task<Analysis?> Think(int simulations, bool hint = false)
    {
        _busy = true;
        _thinking = new CancellationTokenSource();
        Board.AcceptsInput = false;
        StopButton.IsEnabled = true;
        HintButton.IsEnabled = UndoButton.IsEnabled = false;
        StatusText.Text = hint ? $"Analysing the position for {PlayerName(_session.SideToMove)}..."
                               : $"{PlayerName(_session.SideToMove)} (AI) is thinking...";
        int ply = _session.Ply;
        int id = ++_searchId;
        try
        {
            var result = await _session.ThinkAsync(simulations, a => Dispatcher.UIThread.Post(() =>
            {
                // Updates are queued on the UI thread and may arrive after the final result.
                if (id != _searchId || !_busy || _session.Ply != ply) return;
                if (ShowThinking.IsChecked == true) Board.ShowAnalysis(a);
                ShowAnalysisPanel(a, null);
            }), _thinking.Token);
            if (_session.Ply != ply) return null;
            if (ShowThinking.IsChecked == true || hint) Board.ShowAnalysis(result);
            ShowAnalysisPanel(result, null);
            if (hint) StatusText.Text = $"Hint for {PlayerName(_session.SideToMove)}: play {_session.ActionName(result.Moves.MaxBy(m => m.Visits).Action)}";
            return result;
        }
        finally
        {
            _searchId++;
            _busy = false;
            Board.AcceptsInput = true;
            StopButton.IsEnabled = false;
            HintButton.IsEnabled = UndoButton.IsEnabled = true;
        }
    }

    void UpdateStatus()
    {
        Board.Refresh();
        MovesText.Text = _session.Moves.Count == 0 ? "" :
            "Moves: " + string.Join(" ", _session.Moves.Select((m, i) => (i % 2 == 0 ? $"{i / 2 + 1}." : "") + _session.ActionName(m)));
        if (_session.IsOver)
        {
            int w = _session.Winner;
            StatusText.Text = w == 0 ? "Draw." :
                HumanSide == 2 ? $"{PlayerName(w - 1)} wins." :
                w - 1 == HumanSide ? $"You win! ({PlayerName(w - 1)})" : $"The AI wins ({PlayerName(w - 1)}).";
            return;
        }
        string who = PlayerName(_session.SideToMove);
        StatusText.Text = HumanSide == 2 ? $"{who} to move" :
            _session.SideToMove == HumanSide ? $"Your move ({who})" : $"AI to move ({who})";
    }

    void ClearAnalysis()
    {
        MoveList.Children.Clear();
        PvText.Text = "";
        SearchInfo.Text = "";
        EvalLabel.Text = "";
        EvalBar.Width = 0;
    }

    void ShowAnalysisPanel(Analysis a, int? forPlayedMove)
    {
        // The search's value is for the side that was to move when it searched.
        bool aiSearched = forPlayedMove != null || AiToMove;
        double chance = a.WinChance;
        EvalLabel.Text = aiSearched
            ? $"The AI estimates its winning chances at {100 * chance:F0}%."
            : $"Win chance for the side to move: {100 * chance:F0}%.";
        if (EvalBar.Parent is Grid g) EvalBar.Width = Math.Max(0, g.Bounds.Width * chance);
        SearchInfo.Text = $"{a.Simulations:N0} simulations in {a.Seconds:F2}s ({a.Simulations / Math.Max(a.Seconds, 1e-3):N0}/s)";
        PvText.Text = a.PrincipalVariation.Count > 0
            ? "Expected line: " + string.Join("  ", a.PrincipalVariation.Select(_session.ActionName))
            : "";

        MoveList.Children.Clear();
        double total = Math.Max(1, a.TotalVisits);
        foreach (var m in a.Moves.OrderByDescending(m => m.Visits).Take(8))
        {
            var row = new Grid { ColumnDefinitions = new ColumnDefinitions("54,*,58,58") };
            bool best = m.Visits == a.Moves.Max(x => x.Visits);
            row.Children.Add(new TextBlock
            {
                Text = _session.ActionName(m.Action) + (forPlayedMove == m.Action ? "  <" : ""),
                FontWeight = best ? FontWeight.SemiBold : FontWeight.Normal, VerticalAlignment = VerticalAlignment.Center,
            });
            var barHost = new Grid { Margin = new Avalonia.Thickness(0, 3, 8, 3) };
            barHost.Children.Add(new Border { Background = new SolidColorBrush(Color.Parse("#f1f3f5")), CornerRadius = new Avalonia.CornerRadius(4) });
            var bar = new Border
            {
                Background = new SolidColorBrush(Color.Parse(best ? "#e8590c" : "#ffa94d")), CornerRadius = new Avalonia.CornerRadius(4),
                HorizontalAlignment = HorizontalAlignment.Left, Width = 0,
            };
            double share = m.Visits / total;
            barHost.Children.Add(bar);
            barHost.Children.Add(new TextBlock { Text = $" {100 * share:F1}%", FontSize = 11, VerticalAlignment = VerticalAlignment.Center });
            barHost.SizeChanged += (_, e) => bar.Width = e.NewSize.Width * share;
            Grid.SetColumn(barHost, 1);
            row.Children.Add(barHost);
            var prior = new TextBlock { Text = $"{100 * m.Prior:F1}%", HorizontalAlignment = HorizontalAlignment.Right, VerticalAlignment = VerticalAlignment.Center, Foreground = Brushes.Gray };
            Grid.SetColumn(prior, 2);
            row.Children.Add(prior);
            var q = new TextBlock
            {
                Text = float.IsNaN(m.Q) ? "-" : $"{50 * (m.Q + 1):F0}%", HorizontalAlignment = HorizontalAlignment.Right,
                VerticalAlignment = VerticalAlignment.Center,
            };
            Grid.SetColumn(q, 3);
            row.Children.Add(q);
            MoveList.Children.Add(row);
        }
    }
}
