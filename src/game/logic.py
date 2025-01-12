from functools import reduce
from itertools import cycle
import operator
import random
from typing import Dict, List, Set
from src.types.tiles import *


max_all = 356
max_one = 89


class GameContext:
    def __init__(
        self,
        players: list[SquareColor],
        game_state: Dict[SquareColor, int],
    ):
        self.players: list[SquareColor] = players
        self.player_iter: cycle[SquareColor] = cycle(players)
        self.game_state: Dict[SquareColor, int] = game_state
        self.available_pieces: Dict[SquareColor, Set[PieceType]] = {
            player: set(PieceType).difference(extract_pieces(game_state[player]))
            for player in players
        }


def log_available_tiles(context: GameContext):
    print(
        *("{:<6}\t{}".format(k, v) for k, v in context.available_pieces.items()),
        sep="\n",
        end="\n\n",
    )


def log_player_score(context: GameContext):
    print(
        *(
            "{:<6}\t{}".format(k, sum(map(lambda p: np.sum(p.decode()), v)))
            for k, v in context.available_pieces.items()
        ),
        sep="\n",
        end="\n\n",
    )


@lru_cache
def check_valid_step(player_board: int, combined_board: int, step: int) -> bool:
    """check if the current step fits in the game board"""

    @lru_cache
    def __rules(player_board: int, combined_board: int):
        left_fix = combined_board & int("10000000000000000000" * 20, 2)
        right_fix = combined_board & int("00000000000000000001" * 20, 2)
        left = (combined_board - left_fix) << 1
        right = (combined_board - right_fix) >> 1
        top = combined_board << 20
        bottom = combined_board >> 20
        negative_positions = left | right | top | bottom | combined_board

        positive_positions = 0
        if player_board == 0:
            positive_positions = int(
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
            positive_positions = top_left | top_right | bottom_left | bottom_right
            positive_positions -= positive_positions & negative_positions
        return negative_positions, positive_positions

    rules = __rules(player_board, combined_board)
    blocked = step & rules[0]
    contacted = step & rules[1]
    return not blocked and contacted


def step_random(context: GameContext):
    """apply a step following random"""
    player = next(context.player_iter)
    pieces = list(context.available_pieces[player])
    if len(pieces) == 0:
        print("No more tiles for", player)
        return

    player_board = context.game_state[player]
    combined_board = reduce(operator.or_, context.game_state.values())

    piece_cache = PieceType.get_cache()
    choices = [
        (piece, choice)
        for piece in pieces
        for choice in piece_cache[piece].values()
        if check_valid_step(player_board, combined_board, choice)
    ]

    if len(choices) != 0:
        piece_choice = random.choice(choices)
        context.available_pieces[player].discard(piece_choice[0])
        context.game_state[player] |= piece_choice[1]
    else:
        print("No more possible steps for", player)


def step_greedy(context: GameContext):
    player = next(context.player_iter)
    pieces = list(context.available_pieces[player])
    if len(pieces) == 0:
        print("No more tiles for", player)
        return

    player_board = context.game_state[player]
    combined_board = reduce(operator.or_, context.game_state.values())

    piece_cache = PieceType.get_cache()
    choices = [
        (piece, choice)
        for piece in pieces
        for choice in piece_cache[piece].values()
        if check_valid_step(player_board, combined_board, choice)
    ]
    random.shuffle(choices)

    if len(choices) != 0:
        piece_choice = max(choices, key=lambda x: np.sum(x[0].decode()))
        context.available_pieces[player].discard(piece_choice[0])
        context.game_state[player] |= piece_choice[1]
    else:
        print("No more possible steps for", player)
    pass


def __evaluate(context: GameContext) -> int:
    """evaluate the game state for the current player"""
    return [
        np.sum(decode_bitboard(context.game_state[player]))
        for player in context.players
    ]


def __maxn(
    context: GameContext,
    depth: int,
    player_index: int,
):
    if depth <= 0 or all(
        len(pieces) == 0 for pieces in context.available_pieces.values()
    ):
        return __evaluate(context)

    player = list(context.players)[player_index]
    pieces = list(context.available_pieces[player])
    player_board = context.game_state[player]
    combined_board = reduce(operator.or_, context.game_state.values())

    piece_cache = PieceType.get_cache()
    choices = [
        (piece, choice)
        for piece in pieces
        for choice in piece_cache[piece].values()
        if check_valid_step(player_board, combined_board, choice)
    ]
    # size limits
    choices[:5].sort(key=lambda x: np.sum(x[0].decode()))
    choices = choices[:5]

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

        if value[player_index] >= max_one or sum(value) >= max_all:
            return value
        if value[player_index] > best_value[player_index]:
            best_value = value
    return best_value


def step_maxn(context: GameContext):
    player = next(context.player_iter)
    pieces = list(context.available_pieces[player])
    if len(pieces) == 0:
        print("No more tiles for", player)
        return

    player_index = context.players.index(player)
    player_board = context.game_state[player]
    combined_board = reduce(operator.or_, context.game_state.values())

    piece_cache = PieceType.get_cache()
    choices = [
        (piece, choice)
        for piece in pieces
        for choice in piece_cache[piece].values()
        if check_valid_step(player_board, combined_board, choice)
    ]
    # size limits
    choices[:5].sort(key=lambda x: np.sum(x[0].decode()))
    choices = choices[:5]

    best_choice = None
    best_value = [-float("inf")] * len(context.players)
    for choice in choices:
        context.game_state[player] |= choice[1]
        context.available_pieces[player].discard(choice[0])
        value = __maxn(
            context,
            4,
            (player_index + 1) % len(context.players),
        )
        context.game_state[player] &= ~choice[1]
        context.available_pieces[player].add(choice[0])

        if value[player_index] >= max_one or sum(value) >= max_all:
            best_value = value
            best_choice = choice
            break
        if value[player_index] > best_value[player_index]:
            best_value = value
            best_choice = choice

    if best_choice != None:
        context.available_pieces[player].discard(best_choice[0])
        context.game_state[player] |= best_choice[1]
    else:
        print("No more possible steps for", player)
