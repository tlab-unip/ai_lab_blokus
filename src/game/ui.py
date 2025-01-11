from functools import reduce
from itertools import cycle
import operator
import random
from typing import Callable, Dict, Set
from matplotlib import pyplot as plt
import numpy as np
from src.types.tiles import *


class GameContext:
    def __init__(
        self,
        players: list[SquareColor],
        game_state: Dict[SquareColor, int],
    ):
        self.player_iter: cycle[SquareColor] = cycle(players)
        self.game_state: Dict[SquareColor, int] = game_state
        self.available_tiles: Dict[SquareColor, Set[PieceType]] = dict(
            (color, set(PieceType).difference(extract_pieces(game_state.get(color))))
            for color in game_state.keys()
        )


def __render(
    color_dict: Dict[SquareColor, int],
    shape=(20, 20),
) -> np.ndarray:
    board = np.full((*shape, 3), fill_value=240, dtype=int)
    for color, bitboard in color_dict.items():
        decoded_board = decode_bitboard(bitboard, shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if decoded_board[i, j] == 1:
                    board[i, j] = color.toColor()
    return board


def render_pyplot(
    context: GameContext,
    key_step_map: Dict[str, Callable],
):
    fig, ax = plt.subplots()
    board: np.ndarray

    def update_board():
        nonlocal board
        board = __render(context.game_state)
        ax.imshow(board)
        ax.axis("off")
        fig.canvas.draw()

    def on_key(event):
        caller = key_step_map.get(event.key)
        if caller != None:
            caller(context)
            update_board()

    fig.canvas.mpl_connect("key_press_event", on_key)
    update_board()
    plt.show()


def render_pygame(
    context: GameContext,
    key_step_map: Dict[str, Callable],
):
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    clock = pygame.time.Clock()

    def draw_board():
        screen.fill((255, 255, 255))
        board = __render(context.game_state)
        for i in range(20):
            for j in range(20):
                pygame.draw.rect(screen, board[i][j], (j * 20, i * 20, 20, 20))
        pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                for key in key_step_map:
                    if keys[pygame.key.key_code(key)]:
                        key_step_map[key](context)
                        draw_board()
        clock.tick(30)
        draw_board()
    pygame.quit()


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
        negative_positions = left | right | top | bottom

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
    pieces = list(context.available_tiles[player])
    if len(pieces) == 0:
        print("No more tiles for", player)
        return

    player_board = context.game_state[player]
    combined_board = reduce(operator.or_, context.game_state.values())

    piece_cache = PieceType.get_cache()
    piece_choices = [
        (p, step)
        for (p, steps) in ((piece, piece_cache[piece].values()) for piece in pieces)
        for step in steps
        if check_valid_step(player_board, combined_board, step)
    ]
    if len(piece_choices) != 0:
        piece_choice = random.choice(piece_choices)
        context.available_tiles[player].discard(piece_choice[0])
        context.game_state[player] |= piece_choice[1]
    else:
        print("No more possible steps for", player)


if __name__ == "__main__":
    players = [
        SquareColor.RED,
        SquareColor.GREEN,
        SquareColor.BLUE,
        SquareColor.YELLOW,
    ]
    # game_state = {
    #     SquareColor.RED: int("11110000000000000000".ljust(400, "0"), 2),
    #     SquareColor.GREEN: int("00000000000000001111".ljust(400, "0"), 2),
    #     SquareColor.BLUE: int("00000000000000001111".rjust(400, "0"), 2),
    #     SquareColor.YELLOW: int("11110000000000000000".rjust(400, "0"), 2),
    # }
    game_state = dict((color, 0) for color in players)

    key_step_map: Dict[str, Callable[[GameContext], None]] = {
        " ": step_random,
        "e": lambda ctx: print(
            *("{:<6}\t{}".format(k, v) for k, v in ctx.available_tiles.items()),
            sep="\n",
            end="\n\n",
        ),
        "x": lambda ctx: print(
            *(
                "{:<6}\t{}".format(k, sum(map(lambda p: np.sum(p.decode()), v)))
                for k, v in ctx.available_tiles.items()
            ),
            sep="\n",
            end="\n\n",
        ),
    }
    # render_pygame(GameContext(players, game_state), key_step_map)
    render_pyplot(GameContext(players, game_state), key_step_map)
