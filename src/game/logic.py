from functools import reduce, wraps
import operator
import random
from typing import Callable, Dict, Iterable, Set
from src.types.tiles import *
from threading import Thread


max_all = 356
max_one = 89


class GameContext:
    def __init__(
        self,
        players: list[SquareColor],
        game_state: Dict[SquareColor, int],
        time_limit: float = 5,
    ):
        self.players: list[SquareColor] = players
        self.player_index: int = 0
        self.game_state: Dict[SquareColor, int] = game_state
        self.available_pieces: Dict[SquareColor, Set[PieceType]] = {
            player: set(PieceType).difference(extract_pieces(game_state[player]))
            for player in players
        }
        self.running = True
        self.time_limit = time_limit

    @property
    def current_player(self) -> SquareColor:
        return self.players[self.player_index]

    def step_to_next_player(self):
        self.player_index = (self.player_index + 1) % len(self.players)


def __stepper(func):
    """decorator for updating player and timeout handling"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        context: GameContext = args[0]
        context.running = True
        t = Thread(target=func, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
        t.join(context.time_limit)
        if t.is_alive():
            context.running = False
            t.join()
        context.step_to_next_player()

    return wrapper


def log_available_tiles(context: GameContext):
    print(
        *("{:<6}\t{}".format(k, v) for k, v in context.available_pieces.items()),
        sep="\n",
        end="\n\n",
    )


def log_player_score(context: GameContext):
    scores = {
        player: bin(context.game_state[player]).count("1") for player in context.players
    }
    print(
        *("{:<6}\t{}".format(k, v) for k, v in scores.items()),
        sep="\n",
        end="\n\n",
    )


@lru_cache
def __rules(player_board: int, combined_board: int):
    left_fix = combined_board & int("10000000000000000000" * 20, 2)
    right_fix = combined_board & int("00000000000000000001" * 20, 2)
    left = (combined_board - left_fix) << 1
    right = (combined_board - right_fix) >> 1
    top = combined_board << 20
    bottom = combined_board >> 20
    negatives = left | right | top | bottom | combined_board

    positives = 0
    if player_board == 0:
        positives = int(
            "10000000000000000001"
            + "00000000000000000000" * 18
            + "10000000000000000001",
            2,
        )
    else:
        left_fix = player_board & int("10000000000000000000" * 20, 2)
        right_fix = player_board & int("00000000000000000001" * 20, 2)
        top_left = (player_board - left_fix) << 21
        top_right = (player_board - right_fix) << 19
        bottom_left = (player_board - left_fix) >> 19
        bottom_right = (player_board - right_fix) >> 21
        positives = top_left | top_right | bottom_left | bottom_right
        positives &= ~negatives
    return negatives, positives


@lru_cache
def check_valid_step(
    player_board: int,
    combined_board: int,
    step: int,
) -> bool:
    """check if the current step fits in the game board"""
    rules = __rules(player_board, combined_board)
    blocked = step & rules[0]
    contacted = step & rules[1]
    return not blocked and contacted


def __get_choices(
    pieces: Iterable[PieceType],
    player_board: int,
    combined_board: int,
):
    piece_cache = PieceType.get_cache()
    choices = [
        (piece, choice)
        for piece in pieces
        for choice in piece_cache[piece]
        if check_valid_step(player_board, combined_board, choice)
    ]
    random.shuffle(choices)
    return choices


@__stepper
def step_random(context: GameContext):
    """apply a step following random"""
    player = context.current_player
    pieces = tuple(context.available_pieces[player])
    if len(pieces) == 0:
        print("No more tiles for", player)
        return

    player_board = context.game_state[player]
    combined_board = reduce(operator.or_, context.game_state.values())
    choices = __get_choices(pieces, player_board, combined_board)

    if len(choices) != 0:
        piece_choice = random.choice(choices)
        context.available_pieces[player].discard(piece_choice[0])
        context.game_state[player] |= piece_choice[1]
    else:
        print("No more possible steps for", player)


@__stepper
def step_greedy(context: GameContext):
    """get next move following greedy"""
    player = context.current_player
    pieces = tuple(context.available_pieces[player])
    if len(pieces) == 0:
        print("No more tiles for", player)
        return

    player_board = context.game_state[player]
    combined_board = reduce(operator.or_, context.game_state.values())
    choices = __get_choices(pieces, player_board, combined_board)

    best_value = -1
    best_choice = None
    for choice in choices:
        _, positives = __rules(
            player_board | choice[1],
            combined_board | choice[1],
        )
        count = bin(positives).count("1")
        if count > best_value:
            best_value = count
            best_choice = choice

    if best_choice != None:
        context.available_pieces[player].discard(best_choice[0])
        context.game_state[player] |= best_choice[1]
    else:
        print("No more possible steps for", player)

    # if len(choices) != 0:
    #     piece_choice = max(choices, key=lambda x: np.sum(x[0].decode()))
    #     context.available_pieces[player].discard(piece_choice[0])
    #     context.game_state[player] |= piece_choice[1]
    # else:
    #     print("No more possible steps for", player)


def __evaluate(context: GameContext):
    """evaluate by maximizing some value"""
    values = []
    for player in context.players:
        player_board = context.game_state[player]
        combined_board = reduce(operator.or_, context.game_state.values())
        # pieces = tuple(context.available_pieces[player])
        # choices = __get_choices(pieces, player_board, combined_board)
        # values.append(len(choices))
        _, positives = __rules(
            player_board,
            combined_board,
        )
        values.append(bin(positives).count("1"))
    return tuple(values)


@lru_cache
def __maxn(
    context: GameContext,
    depth: int,
    player_index: int,
):
    if depth <= 0 or len(set().union(*context.available_pieces.values())) == 0:
        return __evaluate(context)

    player = context.players[player_index]
    pieces = tuple(context.available_pieces[player])
    player_board = context.game_state[player]
    combined_board = reduce(operator.or_, context.game_state.values())
    choices = __get_choices(pieces, player_board, combined_board)

    best_value = [-float("inf")] * len(context.players)

    for choice in choices:
        context.game_state[player] |= choice[1]
        context.available_pieces[player].discard(choice[0])
        value = __maxn(
            context,
            depth - 1,
            (player_index + 1) % len(context.players),
        )
        context.game_state[player] &= ~choice[1]
        context.available_pieces[player].add(choice[0])

        if value[player_index] > best_value[player_index]:
            best_value = value
        if not context.running:
            break
    return best_value


@__stepper
def step_maxn(
    context: GameContext,
    max_depth: int = 3,
):
    """mxx^n with iterative deepening"""
    player = context.current_player
    pieces = tuple(context.available_pieces[player])

    if len(pieces) == 0:
        print("No more tiles for", player)
        return

    player_index = context.player_index
    player_board = context.game_state[player]
    combined_board = reduce(operator.or_, context.game_state.values())
    choices = __get_choices(pieces, player_board, combined_board)

    best_choice = None
    best_value = [-float("inf")] * len(context.players)

    for depth in range(1, max_depth + 1):
        for choice in choices:
            context.game_state[player] |= choice[1]
            context.available_pieces[player].discard(choice[0])
            value = __maxn(
                context,
                depth,
                (player_index + 1) % len(context.players),
            )
            context.game_state[player] &= ~choice[1]
            context.available_pieces[player].add(choice[0])

            if value[player_index] > best_value[player_index]:
                best_value = value
                best_choice = choice
            if not context.running:
                break
        if not context.running:
            break

    if best_choice != None:
        context.available_pieces[player].discard(best_choice[0])
        context.game_state[player] |= best_choice[1]
    else:
        print("No more possible steps for", player)


def make_step_maxn(depth: int, id=[0]) -> Callable[[GameContext], None]:
    def step_maxn_func(ctx: GameContext):
        step_maxn(ctx, depth)

    id[0] += 1
    step_maxn_func.__name__ = f"step_maxn#{id[0]}_{str(depth)}"
    return step_maxn_func
