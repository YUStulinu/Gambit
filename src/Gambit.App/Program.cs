using Avalonia;

namespace Gambit.App;

static class Program
{
    /// <summary>
    /// Optional command line, mainly for documentation screenshots:
    ///   --game connect4|gomoku   --moves 4453 (or e5,d4,...)   --side 0|1|2   --strength 0..4
    ///   --hint                   analyse the final position
    ///   --screenshot file.png    render the window to a PNG and exit
    /// </summary>
    public static Dictionary<string, string> Options { get; } = new(StringComparer.OrdinalIgnoreCase);

    [STAThread]
    public static void Main(string[] args)
    {
        for (int i = 0; i < args.Length; i++)
            if (args[i].StartsWith("--"))
                Options[args[i][2..]] = i + 1 < args.Length && !args[i + 1].StartsWith("--") ? args[++i] : "true";
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);
    }

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>().UsePlatformDetect().WithInterFont().LogToTrace();
}
