import numpy as np
import pygame
from .env import BlokusEnv
from ..types.tiles import SquareColor


def __extract_observ(
    obs,
    agent_order=(
        SquareColor.BLUE,
        SquareColor.YELLOW,
        SquareColor.RED,
        SquareColor.GREEN,
    ),
):
    color_grid = np.empty((20, 20), dtype=SquareColor)
    for i in range(obs.shape[1]):
        for j in range(obs.shape[2]):
            if obs[0, i, j]:
                color_grid[i, j] = agent_order[0]
            elif obs[1, i, j]:
                color_grid[i, j] = agent_order[1]
            elif obs[2, i, j]:
                color_grid[i, j] = agent_order[2]
            elif obs[3, i, j]:
                color_grid[i, j] = agent_order[3]
            else:
                color_grid[i, j] = SquareColor.EMPTY
    return color_grid


if __name__ == "__main__":
    env = BlokusEnv(render_mode="human")
    env.reset()
    agent_iter = enumerate(env.agent_iter())

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    i, agent = next(agent_iter)
                    observ, reward, termi, trunc, info = env.last()
                    if all([t[1] for t in env.terminations.items()]):
                        continue

                    action_mask = observ["action_mask"]
                    print(action_mask, len(action_mask))
                    action = env.action_space(agent).sample(mask=action_mask)
                    env.step(action)
                    print(action)
                    # obs = env.observe(env.agents[0])["observation"]
                    # colors = __extract_observ(obs)
                    # print(colors)
        pygame.display.flip()
