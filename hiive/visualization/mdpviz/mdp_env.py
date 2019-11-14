import gym
import numpy as np

from hiive.visualization.mdpviz.state import State
from hiive.visualization.mdpviz.transition_probabilities import TransitionProbabilities
from hiive.visualization.mdpviz.action import Action
from hiive.visualization.mdpviz.utils import graph_to_png


class MDPEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'png']}

    def __init__(self, mdp_spec, start_state: State = None):
        self.render_widget = None

        self.mdp_spec = mdp_spec
        self.transitions = TransitionProbabilities(mdp_spec)

        self._previous_state: State = None
        self._previous_action: Action = None
        self._state: State = None
        self._is_done = True
        self.observation_space = gym.spaces.Discrete(self.mdp_spec.num_states)
        self.action_space = gym.spaces.Discrete(self.mdp_spec.num_actions)
        self.start_state = start_state or list(self.mdp_spec.states)[0]

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

    def to_graph(self):
        graph = self.mdp_spec.to_graph(highlight_state=self._previous_state, highlight_action=self._previous_action,
                                       highlight_next_state=self._state)
        return graph

    def render(self, mode='human', close=False):
        if close:
            if self.render_widget:
                self.render_widget.close()
            return

        png_data = graph_to_png(self.to_graph())

        if mode == 'human':
            # TODO: use OpenAI's SimpleImageViewer wrapper when not running in IPython.
            if not self.render_widget:
                from IPython.display import display
                import ipywidgets as widgets

                self.render_widget = widgets.Image()
                display(self.render_widget)

            self.render_widget.value = png_data
        elif mode == 'rgb_array':
            from matplotlib import pyplot
            import io
            return pyplot.imread(io.BytesIO(png_data))
        elif mode == 'png':
            return png_data