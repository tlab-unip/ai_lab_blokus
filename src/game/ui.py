from itertools import cycle
import random
from typing import Callable, Dict
from matplotlib import pyplot as plt
import numpy as np
from src.types.tiles import PieceType, SquareColor, decode_bitboard, encode_bitboard


class GameContext:
    def __init__(self, players, game_state):
        self.player_iter: cycle = cycle(players)
        self.game_state: Dict[SquareColor, int] = game_state


def render_pyplot(
    context: GameContext,
    width=20,
    height=20,
):
    fig, ax = plt.subplots()
    board = np.full((height, width, 3), fill_value=240, dtype=int)
    color_map = {
        SquareColor.RED: [255, 0, 0],
        SquareColor.GREEN: [0, 255, 0],
        SquareColor.BLUE: [0, 0, 255],
        SquareColor.YELLOW: [255, 255, 0],
    }

    def update_board():
        nonlocal board
        board = np.full((height, width, 3), fill_value=240, dtype=int)
        for color, bitboard in context.game_state.items():
            decoded_board = decode_bitboard(bitboard, (height, width))
            for i in range(height):
                for j in range(width):
                    if decoded_board[i, j] == 1:
                        board[i, j] = color_map[color]
        ax.imshow(board)
        ax.axis("off")
        fig.canvas.draw()

    def on_key(event):
        if event.key == "n":
            step_greedy(context)
            update_board()

    fig.canvas.mpl_connect("key_press_event", on_key)
    update_board()
    plt.show()


def render_pygame(
    context,
    key_step_map: Dict[int, Callable],
):
    import pygame

    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    clock = pygame.time.Clock()

    def draw_board():
        screen.fill((255, 255, 255))
        for color, bitboard in context.game_state.items():
            board = decode_bitboard(bitboard)
            for i in range(20):
                for j in range(20):
                    if board[i, j] == 1:
                        pygame.draw.rect(
                            screen, color.toColor(), (j * 20, i * 20, 20, 20)
                        )
        pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                for key in key_step_map:
                    if keys[key]:
                        key_step_map[key](context)
                        draw_board()
        clock.tick(30)
        draw_board()
    pygame.quit()


def step_greedy(context: GameContext):
    pieces = list(PieceType)
    chosen_piece = random.choice(pieces)
    translation = (random.randint(0, 19), random.randint(0, 19))
    rotation = random.randint(0, 3)
    flipping = random.choice([True, False])

    piece_board = chosen_piece.decode(translation, rotation, flipping)
    piece_bitboard = encode_bitboard(piece_board)
    player = next(context.player_iter)

    combined_board = 0
    for bitboard in context.game_state.values():
        combined_board |= bitboard
    if not combined_board & piece_bitboard:
        context.game_state[player] |= piece_bitboard


if __name__ == "__main__":
    players = [
        SquareColor.RED,
        SquareColor.GREEN,
        SquareColor.BLUE,
        SquareColor.YELLOW,
    ]
    game_state = {
        SquareColor.RED: int(
            "11110000000000000000".ljust(400, "0"),
            2,
        ),
        SquareColor.GREEN: int(
            "00000000000000001111".ljust(400, "0"),
            2,
        ),
        SquareColor.BLUE: int(
            "00000000000000001111".rjust(400, "0"),
            2,
        ),
        SquareColor.YELLOW: int(
            "11110000000000000000".rjust(400, "0"),
            2,
        ),
    }

    key_step_map: Dict[int, Callable] = {
        pygame.K_RETURN: step_greedy,
    }
    render_pygame(GameContext(players, game_state), key_step_map)
    # render_pyplot(GameContext(players, game_state))
