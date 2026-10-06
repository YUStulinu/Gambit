using Gambit.Core.Games;

namespace Gambit.Tests;

public class ConnectFourTests
{
    [Theory]
    [InlineData("1212121", true)]     // vertical, column 1
    [InlineData("1122334", true)]     // horizontal along the bottom
    [InlineData("12234334544", true)] // diagonal /: (1,0) (2,1) (3,2) (4,3)
    [InlineData("76654554344", true)] // diagonal \: the mirror image
    [InlineData("1223433454", false)]
    [InlineData("121212", false)]
    public void Detects_four_in_a_row(string moves, bool won)
    {
        var g = Initial(moves[..^1]);
        Assert.Equal(Outcome.Ongoing, g.Outcome);
        g.Play(moves[^1] - '1');
        Assert.Equal(won ? Outcome.Loss : Outcome.Ongoing, g.Outcome);
    }

    static ConnectFour Initial(string moves) => ConnectFour.FromMoves(moves);

    [Fact]
    public void Lines_do_not_wrap_between_columns()
    {
        // Rows 3-5 of column 1 and row 0 of column 2 would be four consecutive cells if the
        // board wrapped; the sentinel bit between columns keeps them apart.
        ulong bits = (0b111UL << 3) | (1UL << 7);
        Assert.False(ConnectFour.HasFour(bits));
        Assert.True(ConnectFour.HasFour(0b1111UL << 2));   // rows 2-5 of column 1: a real vertical four
    }

    [Fact]
    public void Full_board_without_four_is_a_draw()
    {
        // Play random games until one fills the board: it must end as a draw at move 42.
        var rng = new Random(11);
        Span<int> moves = stackalloc int[7];
        for (int game = 0; game < 100_000; game++)
        {
            var g = ConnectFour.Initial;
            while (!g.Outcome.IsOver())
            {
                int n = g.LegalMoves(moves);
                g.Play(moves[rng.Next(n)]);
            }
            if (g.Ply == 42 && g.Outcome == Outcome.Draw) return;
            Assert.Equal(Outcome.Loss, g.Outcome);
        }
        Assert.Fail("no drawn game found");
    }

    [Fact]
    public void Full_column_is_illegal()
    {
        var g = ConnectFour.FromMoves("444444");
        Assert.False(g.IsLegal(3));
        Span<int> moves = stackalloc int[7];
        Assert.Equal(6, g.LegalMoves(moves));
    }

    [Fact]
    public void Cells_and_encoding_match_the_moves()
    {
        var g = ConnectFour.FromMoves("44");
        Assert.Equal(1, g.CellAt(3, 5));   // first player's stone at the bottom (y = 5)
        Assert.Equal(2, g.CellAt(3, 4));
        var planes = new float[3 * 42];
        g.Encode(planes);                   // first player to move again: own = first player's stones
        Assert.Equal(1f, planes[5 * 7 + 3]);
        Assert.Equal(1f, planes[42 + 4 * 7 + 3]);
        Assert.Equal(0f, planes[84]);       // side-to-move plane is zero for the first player
        g.Play(0);
        g.Encode(planes);
        Assert.Equal(1f, planes[42 + 5 * 7 + 3]); // now the first player's stones are "opponent"
        Assert.Equal(1f, planes[84]);
    }

    [Fact]
    public void Mirror_symmetry_maps_cells_and_actions_consistently()
    {
        var g = ConnectFour.FromMoves("1123");
        var m = ConnectFour.FromMoves("7765");
        for (int y = 0; y < 6; y++)
            for (int x = 0; x < 7; x++)
            {
                int cell = ConnectFour.TransformCell(y * 7 + x, 1);
                Assert.Equal(g.CellAt(x, y), m.CellAt(cell % 7, cell / 7));
            }
        Assert.Equal(6, ConnectFour.TransformAction(0, 1));
    }

    [Fact]
    public void Winning_move_detection()
    {
        var g = ConnectFour.FromMoves("121212");
        Assert.True(g.IsWinningMove(0));
        Assert.False(g.IsWinningMove(1));
    }

    [Fact]
    public void Hash_identifies_transpositions()
    {
        Assert.Equal(ConnectFour.FromMoves("1234").Hash, ConnectFour.FromMoves("3214").Hash);
        Assert.NotEqual(ConnectFour.FromMoves("1234").Hash, ConnectFour.FromMoves("1243").Hash);
    }
}

public class GomokuTests
{
    static Gomoku Play(params (int x, int y)[] cells)
    {
        var g = Gomoku.Initial;
        foreach (var (x, y) in cells) g.Play(y * Gomoku.Size + x);
        return g;
    }

    [Theory]
    [InlineData(1, 0)]
    [InlineData(0, 1)]
    [InlineData(1, 1)]
    [InlineData(1, -1)]
    public void Five_in_a_row_wins_in_every_direction(int dx, int dy)
    {
        var g = Gomoku.Initial;
        int x0 = 2, y0 = dy < 0 ? 6 : 2;
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(Outcome.Ongoing, g.Outcome);
            g.Play((y0 + i * dy) * 9 + x0 + i * dx);            // black
            if (i < 4) g.Play(8 * 9 + i);                        // white on the bottom row
        }
        Assert.Equal(Outcome.Loss, g.Outcome);
    }

    [Fact]
    public void Four_is_not_enough_and_lines_do_not_wrap()
    {
        // Black: (6..8, row 0) and (0..1, row 1): five consecutive bits once rows are joined, not a line.
        var g = Play((6, 0), (0, 8), (7, 0), (1, 8), (8, 0), (2, 8), (0, 1), (4, 8), (1, 1));
        Assert.Equal(Outcome.Ongoing, g.Outcome);
    }

    [Fact]
    public void Symmetries_are_bijections_of_the_board()
    {
        for (int s = 0; s < 8; s++)
        {
            var seen = new HashSet<int>();
            for (int c = 0; c < 81; c++) Assert.True(seen.Add(Gomoku.TransformCell(c, s)));
        }
        Assert.Equal(40, Gomoku.TransformCell(40, 5)); // the centre is fixed
    }

    [Fact]
    public void Action_names_round_trip()
    {
        var g = Gomoku.Initial;
        for (int a = 0; a < 81; a++) Assert.Equal(a, g.ParseAction(g.ActionName(a)));
        Assert.Equal("e5", g.ActionName(40));
    }
}
