# Gambit

**An AlphaZero-style game AI in C#.** It starts with no human games, no opening book and no hand-written evaluation. It learns Connect Four (and 9×9 Gomoku) by playing hundreds of thousands of games against itself, and its strength is measured as it grows, against fixed opponents and against a perfect solver.

![Gambit desktop app](docs/app-connect4.png)

How it is built:
- **Search:** Monte Carlo Tree Search guided by a two-headed residual network, which proposes moves and judges positions. Hundreds of games run in parallel, and their leaf evaluations go to the GPU in large batches.
- **Training:** in C# with [TorchSharp](https://github.com/dotnet/TorchSharp) (PyTorch's engine from .NET).
- **Desktop app:** written with Avalonia, it runs the trained network in plain C# with SIMD and no libtorch. While you play, it shows which moves the AI is considering and how sure it is of each.

## Results: Connect Four

One training run: 80 generations, **327,680 self-play games**, 2.7 million training positions, **77 minutes on an RTX 2060**, starting from random weights.

![Training progress](docs/progress-connect4.svg)

### Graded by the perfect solver

The test set is 1000 positions, each with a real decision (see below). A *blunder* is a move that gives away a won or drawn position.

| player | blunders | value sign correct |
|---|---:|---:|
| random move | 51.5% | — |
| classic MCTS, 200 random rollouts | 24.6% | 40.1% |
| classic MCTS, 2,000 random rollouts | 15.7% | 48.7% |
| classic MCTS, 20,000 random rollouts | 11.8% | 57.1% |
| Gambit, untrained network + MCTS 200 | 31.6% | 12.7% |
| Gambit, final network alone (no search) | 14.0% | 78.1% |
| **Gambit, final network + MCTS 200** | **6.1%** | **87.5%** |

With 200 simulations the trained network blunders **half as often as classic MCTS with 20,000 rollouts**, 100× the search. Even the network alone, with no search at all, plays like 2,000 rollouts.

### Against the perfect player

Games start from 10 random moves, and each opening is played twice with colours swapped, so a perfect player scores exactly 50%.

| generation | 0 | 5 | 15 | 30 | 50 | 71 (final) |
|---|---:|---:|---:|---:|---:|---:|
| score vs perfect play | 15% | 30% | 38% | 39% | 43% | 40% |
| blunders (MCTS 200) | 31.6% | 12.5% | 8.4% | 6.9% | 6.2% | 6.1% |

### Elo

In the round robin every saved network played every other, 100 games per pair, with 4 random opening moves. The scale is anchored at classic MCTS with 200 rollouts = 0.

| player | Elo |
|---|---:|
| generation 0 (random weights) | −195 |
| classic MCTS, 200 rollouts | 0 |
| classic MCTS, 2000 rollouts | 124 |
| generation 11 | 339 |
| generation 21 | 460 |
| generation 33 | 527 |
| generation 47 | 554 |
| generation 61 | 591 |
| **generation 71 (final)** | **597** |

These ratings are compressed by the random openings: some openings are decided before the networks play, and strong players draw each other a lot. The final network beats classic MCTS with 2000 rollouts 95–5 in direct games.

### Speed (RTX 2060, 6×64 network)

| | |
|---|---|
| move generator (bitboards, one thread) | 40 M moves/s |
| network on the GPU, fp16, batch 1024 | 123,000 positions/s |
| self-play, 2048 games in parallel | up to 196 games/s, 530,000 simulations/s |
| the same network in plain C# (one thread / all 12 threads) | 780 / 7,000 positions/s |
| the network in the app (search with batches of 16) | 1,400–2,400 simulations/s |

## Results: Gomoku 9×9

40 generations, **81,920 self-play games, 66 minutes**, same network size. Gomoku has no solver, so it is measured by Elo only:

| player | round-robin Elo |
|---|---:|
| generation 0 (random weights) | −331 |
| classic MCTS, 200 rollouts | 0 |
| classic MCTS, 2000 rollouts | 183 |
| generation 10 | 312 |
| generation 20 | 640 |
| generation 30 | 883 |
| **generation 40 (final)** | **1199** |

The curve was still rising steeply when the run stopped, so longer training would keep paying off.

### Both games against classic MCTS with 100× more search

Gambit used 200 simulations per move, classic MCTS 20,000 random rollouts; 50 games each.

| game | result | score |
|---|---|---:|
| Connect Four | +45 −5 | 90% |
| Gomoku 9×9 | +41 −9 | 82% |

![Gomoku training progress](docs/progress-gomoku.svg)

## How it learns

```
           ┌────────────────────────── generation g ───────────────────────────┐
 best net ─┤ 1. self-play   4096 games, MCTS with 200 simulations per move     │
           │ 2. training    the trained net learns from the recent positions   │
           │ 3. gating      candidate vs best, 200 games: it replaces the best │──> best net
           │                only if it is better with 95% confidence (LOS)     │
           │ 4. rating      Elo update; every 5 generations: matches against   │
           │                fixed opponents and grading by the perfect solver  │
           └───────────────────────────────────────────────────────────────────┘
```

**The network** has an input, a tower and two heads:
- **Input:** 3 planes (own stones, opponent stones, side to move).
- **Tower:** a 3×3 convolution, then 6 residual blocks of 64 channels.
- **Policy head:** a probability for every move.
- **Value head:** the expected result in [−1, 1].

**The search.** MCTS with PUCT selection, as in AlphaZero, plus a few refinements:
- **First-play urgency:** an unvisited move starts at the parent's value minus a small reduction, so the search widens only when it has a reason to.
- **Dirichlet noise** at the root during self-play.
- **Tree reuse:** the subtree of the move played is kept between moves and compacted.
- **Virtual loss:** several leaves of one tree can be evaluated in one batch.

**Self-play throughput.** This is what limits AlphaZero, so most of the engineering went here:
- **Bitboards.** Connect Four uses two 64-bit integers and finds a win with four shift-and-AND pairs. Gomoku uses `UInt128` bitboards with a sentinel column. The move generator runs at 40M moves/s on one thread.
- **Game-level batching.** 2048 games advance in lock-step. Each step:
  1. all trees select a leaf, in parallel on the CPU;
  2. one GPU call evaluates them all, in fp16, with the batch padded to a fixed size so cuDNN doesn't plan a new shape at every call;
  3. all trees expand and back up, in parallel.

  A lock-free cache of evaluations skips about a third of the network calls, because every game starts from the same position and openings repeat.
- **Playout cap randomisation** ([KataGo](https://arxiv.org/abs/1902.10565)). 25% of moves get a full 200-simulation search and become training examples. The rest are played with 40 simulations and recorded nowhere, so games finish ~3× faster without weakening the targets.
- **Symmetry augmentation.** Every training position is used in a random mirror image (and in Gomoku, any of the 8 symmetries of the square).
- **Random openings.** Every self-play game starts with 0–4 uniformly random moves (see *Lessons* below for why).

With all this, an RTX 2060 plays about **110 games/s ≈ 190,000 simulations/s** of Connect Four self-play.

**Gating and Elo.** After training, a candidate network plays the current best in a 200-game match. The match uses random openings, each played twice with colours swapped. The candidate replaces the best only if its *likelihood of superiority* is at least 95%. That figure is Φ((wins − losses) / √(wins + losses)), the standard test in chess-engine development. Unlike AlphaZero's "score ≥ 55%", it stays meaningful when most games between strong players are draws. Elo comes from maximum likelihood (the Bradley–Terry model, by Hunter's MM algorithm) over all the games played. The scale is anchored at a fixed opponent: classic MCTS with random rollouts and 200 simulations, rated 0.

Gating matches are a chain in which each network meets only its predecessor, which tends to inflate the ratings. So the final curve also comes from a **round-robin tournament** between saved generations, where every network meets every other.

**Measured against perfect play.** Connect Four is a solved game, so Gambit includes a perfect solver to grade the AI exactly. It is a negamax search on bitboards, after Pascal Pons:
- alpha-beta pruning;
- a transposition table;
- ordering moves by the threats they create;
- discarding moves that hand the opponent a win;
- a null-window search that narrows the score by bisection.

The test set holds 1000 positions, 10–30 moves deep, every move scored exactly (`data/connect4-solved.txt`). Trivial positions are excluded: one-move wins, forced blocks, and positions where every move is equally good. On them Gambit reports two numbers:
- **Blunder rate:** among positions that are not already lost, how often the AI's move throws away the win or the draw.
- **Score against the perfect player:** after 10 random opening moves, each opening is played twice with colours swapped, so a perfect player would score exactly 50%.

## Lessons

**The first run collapsed into one drawn game.** The first training run sampled only its first 8 moves in proportion to the visit counts. By generation ~33 the policy had become so sharp that this "sampling" chose the same moves every time:
- 95% of the 4096 self-play games were the same draw (`444444…`, the centre column filled first, then the board);
- the value head learned to predict 0 everywhere;
- every gating match ended at exactly 50%, so no candidate was ever accepted.

The network was not near-perfect: against the solver it scored only 34%. The run is kept in `runs/connect4-v1`.

Two changes fixed it:
1. games start with a few uniformly random moves, so the network faces thousands of different openings and must learn to *win* the won ones, not just hold one line;
2. gating uses the likelihood of superiority instead of a score threshold.

This mirrors what AlphaZero-style projects run into in practice: in a small game, exploration has to be forced explicitly.

**"Optimal move" is the wrong metric for AlphaZero.** The solver's exact score rewards the *fastest* win. AlphaZero maximises its probability of winning and is indifferent to a slower, safer win. So the fraction of exactly-optimal moves stalled at ~70% while the network kept getting stronger. The fair measure is the blunder rate: how often a move gives away a win or a draw.

## The desktop app

`src/Gambit.App` (Avalonia, Windows / Linux / macOS):

- **What you can choose:**
  - the game (Connect Four or 9×9 Gomoku);
  - your colour, or both sides for analysis;
  - the AI's strength, from the raw network (1 simulation) to 6400 simulations.
- **Live view of the search:**
  - **Above each column (or on each cell):** the share of the simulations spent on that move, updated as the AI thinks. AlphaZero's search spends its time on the moves it believes in.
  - **Win-chance bar:** the AI's estimate of its own winning chances.
  - **Move table:** the moves considered, each with its search share, its *prior* (the network's first impression before searching) and its win estimate.
  - **Expected line:** the sequence of moves the AI expects.
- **Buttons:** *Hint* analyses the position for you, *Undo* takes back a move, and *Move now* stops the search.

The networks ship in `models/` as `.gnet` files of about 2 MB. They run in plain C#: the batch norms are folded into the convolutions at export time, and an AVX2/FMA kernel computes 64 output channels per pass. A test checks this engine against a direct implementation, and the benchmark checks it against TorchSharp; they agree to 2·10⁻⁶. So the app needs neither libtorch nor a GPU.

![Gomoku](docs/app-gomoku.png)

## Running it

Requirements:
- the .NET 9 SDK;
- for training, an NVIDIA GPU (TorchSharp downloads libtorch with CUDA, about 2.5 GB, on the first build).

The app and the tests need no GPU.

```bash
# the desktop app (uses models/connect4.gnet and models/gomoku.gnet)
dotnet run --project src/Gambit.App -c Release

# tests (games, search, solver, self-play, arena, Elo, the C# network)
dotnet test tests/Gambit.Tests -c Release

# train from scratch (resumes automatically if interrupted)
dotnet run --project src/Gambit.Cli -c Release -- train --game connect4 --generations 80
dotnet run --project src/Gambit.Cli -c Release -- train --game gomoku --generations 40

# measure a run
gambit eval       --game connect4 --run runs/connect4 --every 5     # blunders, vs perfect play, per generation
gambit tournament --game connect4 --run runs/connect4 --every 10    # round-robin Elo
gambit chart      --game connect4 --run runs/connect4                # runs/connect4/progress.svg

# other tools
gambit bench --game connect4                                         # speed of every component
gambit arena --game connect4 --a models/connect4.gnet --b perfect --games 100
gambit arena --game connect4 --a runs/connect4/models/gen-0040.pt --b rollout:20000
gambit play  --game connect4                                         # console game
gambit solve --game connect4 --moves 4453432                         # exact score of every move
gambit export --game connect4 --model runs/connect4/models/best.pt --out models/connect4.gnet
```

(`gambit` is `dotnet run --project src/Gambit.Cli -c Release --`, or the `gambit.exe` built in `src/Gambit.Cli/bin`.)

`train` options, with the Connect Four defaults:

| option | default | meaning |
|---|---|---|
| `--games` | `4096` | self-play games per generation |
| `--parallel` | `2048` | games played at the same time (batch size) |
| `--sims` / `--fast-sims` | `200` / `40` | simulations for full / fast moves |
| `--full-prob` | `0.25` | share of moves that get a full search and become samples |
| `--blocks` / `--channels` | `6` / `64` | network size |
| `--lr` | `0.02` | SGD learning rate (×0.3 after 60% of the run, ×0.1 after 85%) |
| `--window` | `250000` | replay window (positions) |
| `--reuse` | `4` | how many times each new position is used in training |
| `--gate` | `0.95` | likelihood of superiority a candidate needs to become the best |
| `--random-opening` | `4` | maximum number of random moves at the start of each self-play game |
| `--fp32` | off | self-play inference in fp32 instead of fp16 |

## Project layout

```
src/Gambit.Core      no dependencies beyond .NET: everything except training
  Games/             IGame (static abstract interface members), ConnectFour, Gomoku
  Search/            SearchTree (PUCT, virtual loss, tree reuse), evaluators, players, eval cache
  Training/          SelfPlayEngine (batched, playout cap randomisation), ReplayBuffer (+ symmetries)
  Rating/            Elo (Bradley-Terry MLE), Arena (batched matches between two searchers)
  Solver/            ConnectFourSolver (perfect play), the solved test set and its grading
  Neural/            NativeNetwork: the trained network in C# (.gnet format)
  Session/           a game in progress with an AI, for the app
src/Gambit.Neural    TorchSharp: the ResNet, GPU evaluator, trainer, checkpoints and .gnet export
src/Gambit.Cli       train, eval, tournament, arena, bench, chart, play, export
src/Gambit.App       the Avalonia desktop app
tests/Gambit.Tests   xUnit: rules, solver vs brute force, search tactics, self-play targets,
                     replay files, arena, Elo, C# network vs a direct implementation, sessions
data/                the 1000 solved Connect Four test positions
models/              the trained networks used by the app
```

The game is a type parameter everywhere (`SearchTree<TGame>`, `SelfPlayEngine<TGame>`, …), with `TGame` a small struct that implements `IGame<TGame>` through C# 11's static abstract members. The JIT therefore compiles the search separately for each game, with no virtual calls and no allocations per simulation. Adding a game means writing one struct.

## Publishing to GitHub

```bash
cd Gambit
git init                                   # already done if the folder has a .git directory
git add -A
git commit -m "Gambit: AlphaZero-style self-play AI in C#"

# create an empty repository named "Gambit" on github.com (no README, no .gitignore), then:
git remote add origin https://github.com/YUStulinu/Gambit.git
git branch -M main
git push -u origin main
```

After later changes: `git add -A && git commit -m "..." && git push`.

## License

MIT
