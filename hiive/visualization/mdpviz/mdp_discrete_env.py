import numpy as np

from hiive.visualization import MDPEnv, State


class MDPDiscreteEnv(MDPEnv):
    metadata = {'render.modes': ['human', 'rgb_array', 'png']}

    def __init__(self, mdp_spec, start_state: State = None):

        super().__init__(mdp_spec, start_state)
        """
        P[s][a] == [(probability, nextstate, reward, done), ...]
        """
        self.P = {s: {a: [] for a in range(mdp_spec.num_actions)} for s in range(mdp_spec.num_states)}

    def reset(self):
        self._previous_state = None
        self._previous_action = None
        self._state = self.start_state
        self._is_done = self._state.terminal_state
        return self._state.index

    def step(self, action_index):
        action = self.mdp_spec.actions[action_index]
        self._previous_state = self._state
        self._previous_action = action

        if not self._is_done:
            reward_probs = self.transitions.rewards[self._state, action]
            reward = np.random.choice(list(reward_probs.keys()), p=list(reward_probs.values()))

            next_state_probs = self.transitions.next_states[self._state, action]
            self._state = np.random.choice(list(next_state_probs.keys()), p=list(next_state_probs.values()))
            self._is_done = self._state.terminal_state
        else:
            reward = 0

        return self._state.index, reward, self._is_done, None

    def render(self, mode='human'):
        return self._render(mode, False)