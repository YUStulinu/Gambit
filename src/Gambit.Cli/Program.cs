using Gambit.Cli;

const string Usage = """
    Gambit - an AlphaZero-style player that learns a game from self-play alone.

    usage: gambit <command> --game connect4|gomoku [options]

    commands:
      train         self-play / train / gate loop (see TrainCommand for options)
      arena         match between two networks or players
      eval          Elo against fixed opponents and accuracy against the perfect solver
      export        convert a checkpoint to a .gnet file for the desktop app
      bench         speed of the network, the search, the solver and self-play
      solver-set    build the Connect Four test set of solved positions
      play          play against a network in the console
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
        _ => throw new ArgumentException($"unknown command '{args[0]}'\n\n{Usage}"),
    };
}
catch (ArgumentException e)
{
    Console.Error.WriteLine(e.Message);
    return 2;
}
