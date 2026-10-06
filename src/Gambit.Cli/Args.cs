using System.Globalization;
using Gambit.Core.Games;

namespace Gambit.Cli;

/// <summary>Minimal "--name value" / "--flag" command-line parsing.</summary>
public sealed class Args
{
    readonly Dictionary<string, string?> _values = new(StringComparer.OrdinalIgnoreCase);
    readonly HashSet<string> _used = new(StringComparer.OrdinalIgnoreCase);
    public readonly List<string> Positional = [];

    public Args(IEnumerable<string> args)
    {
        string? pending = null;
        foreach (var a in args)
        {
            if (a.StartsWith("--"))
            {
                if (pending != null) _values[pending] = null;
                pending = a[2..];
                int eq = pending.IndexOf('=');
                if (eq >= 0)
                {
                    _values[pending[..eq]] = pending[(eq + 1)..];
                    pending = null;
                }
            }
            else if (pending != null)
            {
                _values[pending] = a;
                pending = null;
            }
            else Positional.Add(a);
        }
        if (pending != null) _values[pending] = null;
    }

    public string Get(string name, string def)
    {
        _used.Add(name);
        return _values.TryGetValue(name, out var v) && v != null ? v : def;
    }

    public string? GetOrNull(string name)
    {
        _used.Add(name);
        return _values.TryGetValue(name, out var v) ? v : null;
    }

    public int Int(string name, int def) => int.Parse(Get(name, def.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
    public double Double(string name, double def) => double.Parse(Get(name, def.ToString(CultureInfo.InvariantCulture)), CultureInfo.InvariantCulture);
    public float Float(string name, float def) => (float)Double(name, def);

    public bool Flag(string name)
    {
        _used.Add(name);
        if (!_values.TryGetValue(name, out var v)) return false;
        return v == null || v is "1" or "true" or "yes";
    }

    /// <summary>Fails on options nobody asked for (typos).</summary>
    public void CheckUnused()
    {
        var unknown = _values.Keys.Where(k => !_used.Contains(k)).ToList();
        if (unknown.Count > 0) throw new ArgumentException($"unknown option(s): {string.Join(", ", unknown.Select(u => "--" + u))}");
    }
}

/// <summary>Runs generic code for the game named on the command line.</summary>
public interface IGameCommand
{
    int Run<TGame>(Args args) where TGame : unmanaged, IGame<TGame>;
}

public static class Games
{
    public static readonly string[] Names = ["connect4", "gomoku"];

    public static int Dispatch(string game, IGameCommand cmd, Args args) => game.ToLowerInvariant() switch
    {
        "connect4" or "c4" => cmd.Run<ConnectFour>(args),
        "gomoku" => cmd.Run<Gomoku>(args),
        _ => throw new ArgumentException($"unknown game '{game}' (choose {string.Join(" or ", Names)})"),
    };
}
