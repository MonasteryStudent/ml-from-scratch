import numpy as np
import pytest

from ml_from_scratch.reinforcement_learning.value_iteration import (
    bellman_optimality_update,
    bellman_optimality_update_vectorized,
    value_iteration,
)

# Expected immediate rewards r(s, a).
# Rows represent current states [s0, s1].
# Columns represent actions [a0, a1].
REWARDS = np.array([
    [0.0, 1.0],  # Rewards in s0 for a0 and a1
    [2.0, 0.0],  # Rewards in s1 for a0 and a1
])

# Transition probabilities p(s' | s, a).
# Axis 0: current state s
# Axis 1: selected action a
# Axis 2: possible next state s' in the order [s0, s1]
TRANSITIONS = np.array([
    [  # Current state s0
        [1.0, 0.0],  # a0 leads to s0
        [0.0, 1.0],  # a1 leads to s1
    ],
    [  # Current state s1
        [0.0, 1.0],  # a0 leads to s1
        [1.0, 0.0],  # a1 leads to s0
    ],
])


def test_first_update_matches_manual_calculation():
    q_values, policy, new_values = bellman_optimality_update(
        np.zeros(2), REWARDS, TRANSITIONS, gamma=0.5
    )

    assert np.array_equal(q_values, [[0.0, 1.0], [2.0, 0.0]])
    assert np.array_equal(policy, [1, 0])
    assert np.array_equal(new_values, [1.0, 2.0])


def test_second_update_uses_only_previous_values():
    q_values, policy, new_values = bellman_optimality_update(
        np.array([1.0, 2.0]), REWARDS, TRANSITIONS, gamma=0.5
    )

    assert np.array_equal(q_values, [[0.5, 2.0], [3.0, 0.5]])
    assert np.array_equal(policy, [1, 0])
    assert np.array_equal(new_values, [2.0, 3.0])


def test_stochastic_transition_uses_weighted_next_state_values():
    transitions = TRANSITIONS.copy()
    transitions[0, 0] = [0.25, 0.75]

    q_values, _, _ = bellman_optimality_update(
        np.array([4.0, 8.0]), REWARDS, transitions, gamma=0.5
    )

    assert q_values[0, 0] == 0.5 * (0.25 * 4 + 0.75 * 8)


def test_converges_to_optimal_values_and_policy():
    values, policy, history = value_iteration(
        REWARDS, TRANSITIONS, gamma=0.5, tolerance=1e-10
    )

    assert np.allclose(values, [3.0, 4.0], atol=1e-9)
    assert np.array_equal(policy, [1, 0])
    assert np.array_equal(history[0], [0.0, 0.0])
    assert np.array_equal(history[1], [1.0, 2.0])
    assert np.array_equal(history[2], [2.0, 3.0])


def test_initial_guess_does_not_change_solution_or_mutate_input():
    initial_values = np.array([8.0, -3.0])
    values, policy, history = value_iteration(
        REWARDS, TRANSITIONS, gamma=0.5, initial_values=initial_values
    )

    assert np.allclose(values, [3.0, 4.0])
    assert np.array_equal(policy, [1, 0])
    assert np.array_equal(history[0], [8.0, -3.0])
    assert np.array_equal(initial_values, [8.0, -3.0])


def test_vectorized_update_matches_loop_update():
    values = np.array([1.0, 2.0])

    loop_q_values, loop_actions, loop_values = bellman_optimality_update(
        values,
        REWARDS,
        TRANSITIONS,
        gamma=0.5
    )

    vectorized_q_values, vectorized_actions, vectorized_values = (
        bellman_optimality_update_vectorized(
            values,
            REWARDS,
            TRANSITIONS,
            gamma=0.5
        )
    )

    np.testing.assert_allclose(vectorized_q_values, loop_q_values)
    np.testing.assert_array_equal(vectorized_actions, loop_actions)
    np.testing.assert_allclose(vectorized_values, loop_values)