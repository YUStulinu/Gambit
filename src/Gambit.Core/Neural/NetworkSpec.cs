namespace Gambit.Core.Neural;

/// <summary>
/// Shape of the AlphaZero network: a residual tower with a policy head and a value head.
///
///     input planes ── conv3x3 ── BN ── ReLU ── [residual block] x Blocks ──┬── policy head ── logits (ActionCount)
///                                                                          └── value head ── tanh (1)
///     residual block: conv3x3 ── BN ── ReLU ── conv3x3 ── BN ── (+ input) ── ReLU
///     policy head:    conv1x1 (PolicyChannels) ── BN ── ReLU ── fully connected
///     value head:     conv1x1 (ValueChannels) ── BN ── ReLU ── fully connected (ValueHidden) ── ReLU ── fully connected ── tanh
/// </summary>
public sealed record NetworkSpec
{
    public required string Game { get; init; }
    public required int Planes { get; init; }
    public required int Height { get; init; }
    public required int Width { get; init; }
    public required int Actions { get; init; }
    public int Blocks { get; init; } = 6;
    public int Channels { get; init; } = 64;
    public int PolicyChannels { get; init; } = 4;
    public int ValueChannels { get; init; } = 2;
    public int ValueHidden { get; init; } = 64;

    public int Area => Height * Width;

    public static NetworkSpec For<TGame>(int blocks, int channels) where TGame : struct, Games.IGame<TGame> => new()
    {
        Game = TGame.Name, Planes = TGame.PlaneCount, Height = TGame.Height, Width = TGame.Width, Actions = TGame.ActionCount,
        Blocks = blocks, Channels = channels,
    };

    public override string ToString() => $"{Blocks}x{Channels} ResNet";
}
