using Gambit.Cli;

const string Usage = """
    Gambit - an AlphaZero-style player that learns a game from self-play alone.

    usage: gambit <command> --game connect4|gomoku [options]

    commands:
      train         the AlphaZero loop: self-play, training, gating, Elo (resumes an existing run)
      eval          accuracy against the perfect solver (Connect Four) and matches against fixed opponents
      arena         a match between two players: .pt / .gnet networks, rollout:N, perfect
      export        convert a checkpoint (.pt) to a .gnet file for the desktop app
      bench         speed of the move generator, the network, the solver and self-play
      play          play against a network in the console
      tournament    round robin between saved generations: an Elo curve where every network meets every other
      chart         redraw a run's progress chart (progress.svg) from its log

    examples:
      gambit train --game connect4 --generations 80
      gambit eval  --game connect4 --model runs/connect4/models/best.pt --baselines
      gambit arena --game connect4 --a models/connect4.gnet --b rollout:20000 --games 100
    """;

if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
{
    Console.WriteLine(Usage);
    return 0;
}

var a = new Args(args.Skip(1));
string game = a.Get("game", "connect4");
try
{
    return args[0] switch
    {
        "bench" => Games.Dispatch(game, new BenchCommand(), a),
        "train" => Games.Dispatch(game, new TrainCommand(), a),
        "eval" => Games.Dispatch(game, new EvalCommand(), a),
        "arena" => Games.Dispatch(game, new ArenaCommand(), a),
        "export" => Games.Dispatch(game, new ExportCommand(), a),
        "play" => Games.Dispatch(game, new PlayCommand(), a),
        "chart" => Games.Dispatch(game, new ChartCommand(), a),
        "tournament" => Games.Dispatch(game, new TournamentCommand(), a),
        _ => throw new ArgumentException($"unknown command '{args[0]}'\n\n{Usage}"),
    };
}
catch (ArgumentException e)
{
    Console.Error.WriteLine(e.Message);
    return 2;
}
