import typing
from collections import defaultdict

import networkx as nx
import numpy as np

from hiive.visualization.mdpviz.mdp_discrete_env import MDPDiscreteEnv
from hiive.visualization.mdpviz.mdp_env import MDPEnv
from hiive.visualization.mdpviz.transition import Transition
from hiive.visualization.mdpviz.state import State
from hiive.visualization.mdpviz.transition_probabilities import TransitionProbabilities
from hiive.visualization.mdpviz.action import Action
from hiive.visualization.mdpviz.next_state import NextState
from hiive.visualization.mdpviz.reward import Reward
from hiive.visualization.mdpviz.outcome import Outcome


class MDPSpec(object):
    def __init__(self, verbose=False):
        self._states = {}
        self._actions = {}
        self.states = []
        self.actions = []
        self.state_outcomes: typing.Dict[tuple, typing.List[NextState]] = defaultdict(list)
        self.reward_outcomes: typing.Dict[tuple, typing.List[Reward]] = defaultdict(list)
        self.discount = 1.0
        self._node_attribute_dictionary = {}
        self._edge_attribute_dictionary = {}
        self.verbose = verbose

    def state(self, name=None, terminal_state=False):
        if not name:
            if not terminal_state:
                name = 'S%s' % self.num_states
            else:
                name = 'T%s' % self.num_states

        if name not in self.states:
            new_state = State(name, self.num_states, terminal_state=terminal_state)
            self._states[name] = new_state
            self.states.append(new_state)
        return self._states[name]

    def action(self, name=None):
        if not name:
            name = 'A%s' % self.num_actions

        if name not in self.actions:
            new_action = Action(name, self.num_actions)
            self._actions[name] = new_action
            self.actions.append(new_action)
        return self._actions[name]

    def transition(self, state: State, action: Action, outcome: Outcome):
        """Specify either a next state or a reward as `outcome` for a transition."""

        if isinstance(outcome, NextState):
            self.state_outcomes[state, action].append(outcome)
        elif isinstance(outcome, Reward):
            self.reward_outcomes[state, action].append(outcome)
        else:
            raise NotImplementedError()

    @property
    def num_states(self):
        return len(self._states)

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def is_deterministic(self):
        for state in self.states:
            for action in self.actions:
                if len(self.reward_outcomes[state, action]) > 1:
                    return False
                if len(self.state_outcomes[state, action]) > 1:
                    return False
        return True

    def __repr__(self):
        return 'Mdp(states=%s, actions=%s, state_outcomes=%s, reward_outcomes=%s)' % (
            self.states, self.actions, dict(self.state_outcomes), dict(self.reward_outcomes))

    def set_edge_attributes(self, u, v, **kwargs):
        key = (u, v)
        update_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if key not in self._edge_attribute_dictionary:
            self._edge_attribute_dictionary[key] = {}
        self._edge_attribute_dictionary[key].update(update_kwargs)

        del_kwargs = {k: v for k, v in kwargs.items() if v is None}
        for k in del_kwargs:
            self._edge_attribute_dictionary.pop(k, None)

        if len(self._edge_attribute_dictionary[key]) == 0:
            self._edge_attribute_dictionary.pop(key, None)

    def set_node_attributes(self, n, **kwargs):
        update_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if n not in self._node_attribute_dictionary:
            self._node_attribute_dictionary[n] = {}
        self._node_attribute_dictionary[n].update(update_kwargs)

        del_kwargs = {k: v for k, v in kwargs.items() if v is None}
        for k in del_kwargs:
            self._node_attribute_dictionary.pop(k, None)

        if len(self._node_attribute_dictionary[n]) == 0:
            self._node_attribute_dictionary.pop(n, None)

    def to_graph(self, highlight_state: State = None, highlight_action: Action = None,
                 highlight_next_state: State = None):
        transitions = TransitionProbabilities(self)

        graph = nx.MultiDiGraph()
        self._node_attribute_dictionary = {}
        self._edge_attribute_dictionary = {}

        for state in self.states:
            fillcolor = 'yellow' if highlight_state == state else 'red' if highlight_next_state == state else '#FFFFC0'
            self.set_node_attributes(n=state,
                                     shape='doubleoctagon' if state.terminal_state else 'circle',
                                     label=state.name,
                                     fillcolor=fillcolor,
                                     style='filled',
                                     type='state')
            # for s2 in self.states:
            #    self.set_edge_attributes(u=state, v=s2, label=f'{state.name} ->{s2.name}')

        t_index = 0
        for state in self.states:
            if not state.terminal_state:
                for action in self.actions:
                    reward_probs = transitions.rewards[state, action].items()
                    expected_reward = sum(reward * prob for reward, prob in reward_probs)
                    stddev_reward = (sum(
                        reward * reward * prob for reward, prob in
                        reward_probs) - expected_reward * expected_reward) ** 0.5

                    action_label = '%s %+.2f' % (action.name, expected_reward)
                    if len(reward_probs) > 1:
                        action_label += ' (%.2f)' % stddev_reward

                    next_states = transitions.next_states[state, action].items()
                    if len(next_states) == 1:
                        next_state, _ = list(next_states)[0]
                        self.set_edge_attributes(u=state, v=next_state, type='state_to_state', color=color, label=action_label)
                    else:
                        transition = Transition(action, state, t_index)
                        t_index += 1
                        action_color = action.index + 1
                        color = f'/dark28/{action_color}'
                        self.set_edge_attributes(u=state, v=transition, type='state_to_transition', color=color, label=action_label)
                        # transition_label = f'{state.name, action.name}'
                        self.set_node_attributes(n=transition, type='transition', # label=transition_label,
                                                 fillcolor='#FFE0FF', style='filled, bold', shape='point')

                        for next_state, prob in next_states:
                            if not prob:
                                continue
                            color = f'/set28/{action_color}'
                            self.set_edge_attributes(u=transition, v=next_state, label='%3.2f%%' % (prob * 100),
                                                     color=color,
                                                     type='transition_to_state')

                        if state == highlight_state and action == highlight_action:
                            self.set_node_attributes(n=transition, style='bold')
                            self.set_edge_attributes(u=transition, v=next_state, style='bold', color='green')
                            if highlight_next_state:
                                self.set_edge_attributes(u=transition, v=highlight_next_state, style='bold',
                                                         color='red', data='e4')
                        """
                        else:
                            next_state, _ = list(next_states)[0]
                            # if state == highlight_state and action == highlight_action:
                            self.set_node_attributes(n=(next_state, action), label=action_label, style='bold', color='red')

                        """
        # build nodes
        # graph.node.clear()
        for n, node_attributes in self._node_attribute_dictionary.items():
            graph.add_node(node_for_adding=n, **node_attributes)
            if self.verbose:
                print(f'Adding node: {n}, nodes={len(graph.node)}')
        if self.verbose:
            print()

        # build edges
        # graph.edge.clear()
        for edge_key, edge_attributes in self._edge_attribute_dictionary.items():
            u, v = edge_key
            graph.add_edge(u_for_edge=u, v_for_edge=v, **edge_attributes)
            if self.verbose:
                print(f'Adding edge: u={u}, v={v}, edges={len(graph.edge)}, attributes={edge_attributes}')

        if self.verbose:
            print()
        # for some reason, adding edges clears out the node dictionary.
        # so we set it now.
        """
        for n, node_attributes in self._node_attribute_dictionary.items():
            graph.node[n] = node_attributes
            if self.verbose:
                print(f'Setting node attributes for [{n}]: {node_attributes}')
        if self.verbose:
            print()
        """


        return graph

    def get_node_attributes(self, graph, state):
        if isinstance(graph.nodes, dict):
            attributes = graph.nodes[state]
        else:
            attributes = graph.nodes(state)[0][1]
        return attributes

    def to_env(self):
        return MDPEnv(self)

    def to_discrete_env(self):
        return MDPDiscreteEnv(self)

    def validate(self):
        # For now, just validate by trying to compute the transitions.
        # It will raise errors if anything is wrong.
        TransitionProbabilities(self)
        return self

    def get_transition_and_reward_arrays(self, p_default=0.5):
        """Generate the fire management transition and reward matrices.

        The output arrays from this function are valid input to the mdptoolbox.mdp
        classes.

        Let ``S`` = number of states, and ``A`` = number of actions.

        Parameters
        ----------
        p_default : float
            The class-independent probability of the population staying in its
            current population abundance class.

        Returns
        -------
        out : tuple
            ``out[0]`` contains the transition probability matrices P and
            ``out[1]`` contains the reward vector R. P is an  ``A`` × ``S`` × ``S``
            numpy array and R is a numpy vector of length ``S``.

        """
        assert 0 <= p_default <= 1, "'p_default' must be between 0 and 1"
        # The transition probability array
        n_actions = len(self.actions)
        n_states = len(self.states)
        transition_probabilities = np.zeros((n_actions, n_states, n_states))
        # The reward vector
        rewards = np.zeros((n_states, n_actions))
        # Loop over all states
        for state in self.states:
            s = state.index
            # Loop over all actions
            w = 0.0
            total_transition_weight = 0
            for action in self.actions:
                a = action.index
                reward_info = self.reward_outcomes[(state, action)]
                r = np.sum([rwi.outcome * rwi.weight for rwi in reward_info])
                w += np.sum([rwi.weight for rwi in reward_info])
                rewards[s, a] = r
                # Assign the transition probabilities for this state, action pair
                if state.terminal_state:
                    pass
                    transition_probabilities[a][s][s] = 1.0
                    total_transition_weight += 1.0
                else:
                    transitions = self.state_outcomes[(state, action)]
                    total_transition_weight += np.sum([so.weight for so in transitions])
                    for transition in transitions:
                        state_next = transition.outcome
                        transition_probabilities[a][s][state_next.index] = transition.weight
                # transition_probabilities[a, s, ] /= total_transition_weight
                transition_probabilities[a, s, :] /= np.sum(transition_probabilities[a, s, :])
            if w > 0:
                rewards[s, :] /= w

        return transition_probabilities, rewards
