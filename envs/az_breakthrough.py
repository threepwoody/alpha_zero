"""Breakthrough env class."""
from typing import Tuple
import numpy as np
from copy import copy

from envs.base import BoardGameEnv
import sgf_wrapper
from util import get_time_stamp

import momaland
from momaland.utils.aec_wrappers import LinearizeReward
from momaland.envs.breakthrough import mobreakthrough_v0
import gym
from gym.spaces import Box, Discrete


class BreakthroughEnv(BoardGameEnv):
    """Breakthrough Environment with OpenAI Gym api.

    Uses MOMAland's MO-Breakthrough with the LinearizeReward wrapper to make it single-objective.

    """

    def __init__(self, board_size: int = 6, num_stack: int = 1, objective_weights={"player_0": np.array([1, 0, 0, 0]),"player_1": np.array([0, 0, 1, 0])}) -> None:
        """
        Args:
            board_size: board size, default 6
            num_stack: stack last N history states, default 1
        """

        # Breakthrough has no pass move and resign move
        super().__init__(
            id='Breakthrough',
            board_size=board_size,
            num_stack=num_stack,
            has_pass_move=False,
            has_resign_move=False,
        )

        self.objective_weights = objective_weights
        self.env = mobreakthrough_v0.env(board_width=board_size, board_height=board_size, num_objectives=4, render_mode="ansi")
        self.env = LinearizeReward(self.env, objective_weights)
        self.action_dim = self.env.max_move
        self.action_space = Discrete(self.action_dim)
        self.legal_actions = np.ones(self.action_dim, dtype=np.int8).flatten()
        self.total_rewards = {agent: 0 for agent in self.env.agents}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Plays one move."""
        if self.is_game_over():
            raise RuntimeError('Game is over, call reset before using step method.')
        if action is not None and action != self.resign_move and not 0 <= int(action) <= self.action_space.n - 1:
            raise ValueError(f'Invalid action. The action {action} is out of bound.')
        if action is not None and action != self.resign_move and self.legal_actions[int(action)] != 1:
            raise ValueError(f'Illegal action {action}.')

        self.last_move = copy(int(action))
        self.last_player = copy(self.to_play)
        self.steps += 1
        self.add_to_history(self.last_player, self.last_move)

        self.env.step(action)

        self.board = self.env.board
        # Make sure the latest board position is always at index 0
        self.board_deltas.appendleft(np.copy(self.board))
        self.accumulate_total_rewards()
        self.update_winner()
        done = self.is_game_over()
        reward = self.env._cumulative_rewards[self._internal_player(self.to_play)]
        # Switch next player
        self.to_play = self.opponent_player
        self.legal_actions = self.env.observe(self._internal_player(self.to_play))["action_mask"]

        #TODO test observations and compare to env's observations; also, how can we make sure ALL agents are getting their rewards? seems to only work for acting agent here. this is all inheriting from regular gym.Env, so... not sure how they would ever get collected??
        return self.observation(), reward, done, {}

    def update_winner(self):
        _, _, game_over, _, _ = self.env.last()
        if game_over:
            if self.total_rewards[self._internal_player(self.to_play)] >= self.total_rewards[self._internal_player(self.opponent_player)]:
                self.winner = self.to_play
            else:
                self.winner = self.opponent_player
        pass

    def is_game_over(self) -> bool: #TODO debug
        if self.winner is not None:
            return True
        return False

    def get_result_string(self) -> str:
        if not self.is_game_over():
            return ''

        if self.winner == self.black_player:
            return 'B+1.0'
        elif self.winner == self.white_player:
            return 'W+1.0'
        else:
            return 'DRAW'

    def to_sgf(self) -> str:
        return sgf_wrapper.make_sgf(
            board_size=self.board_size,
            move_history=self.history,
            result_string=self.get_result_string(),
            ruleset='',
            komi='',
            date=get_time_stamp(),
        )

    def accumulate_total_rewards(self):
        _, rewards, _, _, _ = self.env.last()
        for agent in self.env.agents:
            self.total_rewards[agent] += rewards[agent]
            print("rewards: ", rewards[agent])
            print("total_rewards: ", self.total_rewards[agent])
        pass

    def reset(self, **kwargs) -> np.ndarray:
        """Reset game to initial state."""
        super().reset(**kwargs)
        self.env.reset()
        self.board = self.env.board
        self.legal_actions = self.env.observe(self._internal_player(self.to_play))["action_mask"]
        return self.observation()

    def is_legal_move(self, move: int) -> bool:
        """Returns bool state to indicate given move is valid or not."""
        if move is None:
            return False
        elif move < 0 or move > self.action_dim - 1:
            return False
        else:
            return self.legal_actions[move] == 1

    def _internal_player(self, player: int) -> str:
        return 'player_'+str(player-1)


if __name__ == '__main__':
    breakthrough = BreakthroughEnv()
    breakthrough.reset()
    done = False

    while not done:
        print(breakthrough.board)
        print(breakthrough.legal_actions)
        legit_actions = np.flatnonzero(breakthrough.legal_actions)
        print(legit_actions)
        action = np.random.choice(legit_actions, 1).item()
        print(action)
        obs, reward, done, _ = breakthrough.step(action)

    # for agent in environment.agent_iter():
    #     action = environment.action_space(agent).seed(42)
    #     observation, reward, termination, truncation, info = environment.last()
    #
    #     print("rewards", environment.rewards)
    #     if termination or truncation:
    #         action = None
    #     else:
    #         if observation:
    #             # this is where you would insert your policy
    #             action = np.where(observation["action_mask"] != 0)[0][0]
    #             print("observation: ", observation)
    #             # print("cumulative rewards", environment._cumulative_rewards)
    #             # print("action: ", action)
    #
    #     environment.step(action)
    #
    # # print("observation: ", observation)
    # print("reward: ", reward)
    # print("rewards", environment.rewards)
    # # print("cumulative rewards", environment._cumulative_rewards)
    # # name = input("Press key to end\n")
    # environment.close()
