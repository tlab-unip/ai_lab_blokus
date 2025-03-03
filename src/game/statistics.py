from typing import Callable, List
from matplotlib import pyplot as plt
import pandas as pd
from src.types.tiles import *
from src.game.logic import *
import contextlib
from tqdm import tqdm
import argparse
import seaborn as sns


def run_simulations(
    players: List[SquareColor],
    steppers: List[Callable[[GameContext], None]],
    initial_game_state: Dict[SquareColor, int],
    num_simulations: int,
    step_time_limit: float = 20,
) -> List[Dict[str, List]]:
    results = []
    try:
        for _ in tqdm(range(num_simulations), desc="Running simulations"):
            state = {player: 0 for player in players}
            state.update(initial_game_state)
            context = GameContext(players, state, step_time_limit)

            last_state = None
            while True:
                if context.current_player == SquareColor.RED:
                    current_state = reduce(operator.or_, context.game_state.values())
                    if last_state != current_state:
                        last_state = current_state
                    else:
                        break
                stepper = steppers[context.player_index]
                with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
                    stepper(context)

            result = {
                "Player": [player.name for player in players],
                "Score": [
                    bin(context.game_state[player]).count("1") for player in players
                ],
                "Algorithm": [stepper.__name__ for stepper in steppers],
            }
            results.append(result)
    except KeyboardInterrupt:
        print("Simulation interrupted. Saving results so far...")
    return results


def compute_statistics(
    results: List[Dict[str, List]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = {"Run": [], "Player": [], "Score": [], "Algorithm": []}
    for i, result in enumerate(results):
        run_number = [i + 1] * len(result["Player"])
        data["Run"].extend(run_number)
        data["Player"].extend(result["Player"])
        data["Score"].extend(result["Score"])
        data["Algorithm"].extend(result["Algorithm"])

    df = pd.DataFrame(data)
    stats = df.groupby("Algorithm")["Score"].describe()
    return df, stats


def visualize_results(
    df: pd.DataFrame,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    sns.boxplot(x="Algorithm", y="Score", data=df, ax=axes[0])
    sns.stripplot(
        x="Algorithm",
        y="Score",
        data=df,
        color="0.3",
        ax=axes[0],
        jitter=True,
    )
    axes[0].set_title("Algorithm Score Distribution")
    axes[0].set_xlabel("Algorithm")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=45)

    df.groupby("Algorithm")["Score"].mean().plot(
        kind="bar", color="skyblue", ax=axes[1]
    )
    axes[1].set_title("Average Score by Algorithm")
    axes[1].set_xlabel("Algorithm")
    axes[1].set_ylabel("Average Score")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run simulations and analyze results.",
    )
    parser.add_argument(
        "--num_simulations",
        "-n",
        type=int,
        default=100,
        help="Number of simulations to run",
    )
    args = parser.parse_args()

    players = [
        SquareColor.RED,
        SquareColor.GREEN,
        SquareColor.BLUE,
        SquareColor.YELLOW,
    ]
    steppers = [
        make_step_maxn(8),
        make_step_maxn(4),
        step_random,
        step_greedy,
    ]
    initial_game_state = {player: 0 for player in players}
    results = run_simulations(
        players,
        steppers,
        initial_game_state,
        args.num_simulations,
    )

    df, stats = compute_statistics(results)
    # df.to_csv("results.csv")
    # stats.to_csv("statistics.csv")
    visualize_results(df)
