using Gambit.Core.Games;
using Gambit.Core.Search;
using Gambit.Core.Session;

namespace Gambit.Tests;

public class SessionTests
{
    [Theory]
    [InlineData("connect4")]
    [InlineData("gomoku")]
    public async Task Thinking_runs_the_requested_simulations(string game)
    {
        var s = Sessions.Create(game, null);
        s.Play(s.ActionAt(s.Width / 2, s.Height - 1));
        var a = await s.ThinkAsync(300, null, CancellationToken.None);
        Assert.True(a.Simulations >= 300, $"only {a.Simulations} simulations");
        Assert.Equal(a.Simulations - 1, a.TotalVisits);
    }

    [Fact]
    public void Undo_and_winning_line()
    {
        var s = Sessions.Create("connect4", null);
        foreach (int c in new[] { 0, 1, 0, 1, 0, 1 }) s.Play(c);
        s.Play(0);
        Assert.True(s.IsOver);
        Assert.Equal(1, s.Winner);
        Assert.Equal(4, s.WinningLine.Count);
        Assert.Equal((0, 2), s.LastCell);
        s.Undo();
        Assert.False(s.IsOver);
        Assert.Equal(6, s.Ply);
    }
}

public class SessionRepro
{
    [Fact]
    public async Task Gomoku_position_from_the_app()
    {
        var s = Sessions.Create("gomoku", null);
        foreach (var name in new[] { "e5", "d4", "e4", "d5", "e6" })
            s.Play(Enumerable.Range(0, 81).First(a => s.IsLegal(a) && s.ActionName(a) == name));
        var a = await s.ThinkAsync(400, null, CancellationToken.None);
        Assert.True(a.Simulations >= 400, $"only {a.Simulations} simulations");
    }
}
