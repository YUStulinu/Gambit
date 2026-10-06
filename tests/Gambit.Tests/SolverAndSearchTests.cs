using Gambit.Core.Games;
using Gambit.Core.Search;
using Gambit.Core.Solver;

namespace Gambit.Tests;

public class SolverTests
{
    /// <summary>Plain negamax over the whole remaining tree, scored like the solver.</summary>
    static int BruteForce(ConnectFour g)
    {
        if (g.Outcome == Outcome.Draw) return 0;
        int best = int.MinValue;
        for (int c = 0; c < 7; c++)
        {
            if (!g.IsLegal(c)) continue;
            var child = g;
            child.Play(c);
            int score = child.Outcome == Outcome.Loss ? (43 - g.Ply) / 2 : -BruteForce(child);
            best = Math.Max(best, score);
        }
        return best;
    }

    static ConnectFour RandomPosition(Random rng, int plies)
    {
        Span<int> moves = stackalloc int[7];
        while (true)
        {
            var g = ConnectFour.Initial;
            while (g.Ply < plies && !g.Outcome.IsOver())
            {
                int n = g.LegalMoves(moves);
                g.Play(moves[rng.Next(n)]);
            }
            if (!g.Outcome.IsOver()) return g;
        }
    }

    [Fact]
    public void Matches_brute_force_on_late_positions()
    {
        var solver = new ConnectFourSolver(16);
        var rng = new Random(3);
        for (int i = 0; i < 60; i++)
        {
            var g = RandomPosition(rng, 30 + i % 6);
            Assert.Equal(BruteForce(g), solver.Solve(g));
        }
    }

    [Fact]
    public void Finds_the_known_values_of_simple_positions()
    {
        var solver = new ConnectFourSolver(18);
        // Three in the first column for the side to move: it wins on the next move.
        var g = ConnectFour.FromMoves("121212");
        Assert.Equal((43 - 6) / 2, solver.Solve(g));
        var scores = solver.ScoreMoves(g);
        Assert.Equal(scores.Max(), scores[0]);
    }

    [Fact]
    public void Solves_a_midgame_position_quickly()
    {
        // A position from Pons' benchmark set, checked against brute force.
        var solver = new ConnectFourSolver(20);
        var g = ConnectFour.FromMoves("2252576253462244111563365343671351441");
        Assert.Equal(BruteForce(g), solver.Solve(g));
    }
}

public class SearchTests
{
    [Fact]
    public void Mcts_takes_an_immediate_win()
    {
        var g = ConnectFour.FromMoves("121212");   // column 1 wins for the side to move
        var player = new MctsPlayer<ConnectFour>(new UniformEvaluator<ConnectFour>(), 200);
        Assert.Equal(0, player.ChooseMove(g));
    }

    [Fact]
    public void Mcts_blocks_an_immediate_loss()
    {
        // The first player has three stones in column 1: the second player must block there.
        var g = ConnectFour.FromMoves("12121");
        var player = new MctsPlayer<ConnectFour>(new RolloutEvaluator<ConnectFour>(4, 1), 400);
        Assert.Equal(0, player.ChooseMove(g));
    }

    [Fact]
    public void Visits_are_consistent_and_the_subtree_survives_advance()
    {
        var player = new MctsPlayer<ConnectFour>(new RolloutEvaluator<ConnectFour>(1, 2), 300, batch: 8);
        var r = player.Search(ConnectFour.Initial, 300);
        Assert.Equal(300, r.Moves.Sum(m => m.Visits) + 1);   // the root's own first visit is its expansion
        var tree = player.Tree;
        int best = r.BestMove;
        int childVisits = r.Moves.First(m => m.Action == best).Visits;
        tree.Advance(best);
        Assert.Equal(childVisits, tree.RootVisits);
        Assert.Equal(1, tree.RootState.Ply);
        Assert.Equal(0, tree.RootNode.Virtual);
    }

    [Fact]
    public void Pure_mcts_beats_random_play()
    {
        var mcts = new MctsPlayer<ConnectFour>(new RolloutEvaluator<ConnectFour>(1, 5), 200);
        var random = new RandomPlayer<ConnectFour>(7);
        int wins = 0;
        for (int game = 0; game < 10; game++)
        {
            var g = ConnectFour.Initial;
            mcts.NewGame();
            bool mctsFirst = game % 2 == 0;
            while (!g.Outcome.IsOver())
            {
                bool mctsTurn = (g.SideToMove == 0) == mctsFirst;
                g.Play(mctsTurn ? mcts.ChooseMove(g) : random.ChooseMove(g));
            }
            bool mctsMovedLast = ((g.Ply - 1) & 1) == (mctsFirst ? 0 : 1);
            if (g.Outcome == Outcome.Loss && mctsMovedLast) wins++;
        }
        Assert.True(wins >= 9, $"pure MCTS won only {wins}/10");
    }
}

public class ChooseMoveTests
{
    /// <summary>Puts all the prior on one action.</summary>
    sealed class OneMoveEvaluator(int action) : IEvaluator<ConnectFour>
    {
        public void Evaluate(ReadOnlySpan<ConnectFour> states, Span<float> logits, Span<float> values)
        {
            for (int i = 0; i < states.Length; i++)
            {
                for (int a = 0; a < 7; a++) logits[i * 7 + a] = a == action ? 5f : 0f;
                values[i] = 0;
            }
        }
    }

    [Fact]
    public void A_single_simulation_plays_the_network_choice()
    {
        var player = new MctsPlayer<ConnectFour>(new OneMoveEvaluator(5), 1);
        Assert.Equal(5, player.ChooseMove(ConnectFour.Initial));
    }
}
