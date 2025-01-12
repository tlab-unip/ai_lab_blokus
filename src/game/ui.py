from typing import Callable, Dict
from matplotlib import pyplot as plt
import numpy as np
from src.types.tiles import *
from src.game.logic import *


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
    game_state = dict((player, 0) for player in players)

    key_step_map: Dict[str, Callable[[GameContext], None]] = {
        "1": step_random,
        "2": step_greedy,
        "3": step_maxn,
        "z": log_available_tiles,
        "x": log_player_score,
    }
    # render_pygame(GameContext(players, game_state), key_step_map)
    render_pyplot(GameContext(players, game_state), key_step_map)
