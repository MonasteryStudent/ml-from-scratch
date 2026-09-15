import numpy as np
import pytest

from ml_from_scratch.reinforcement_learning.value_iteration import (
    bellman_optimality_update,
    value_iteration,
)


# Rows: s0, s1. Columns: a0, a1. Last transition axis: s0, s1.
REWARDS = np.array([[0.0, 1.0], [2.0, 0.0]])
TRANSITIONS = np.array([
    [[1.0, 0.0], [0.0, 1.0]],
    [[0.0, 1.0], [1.0, 0.0]],
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


def test_rejects_invalid_transition_distribution():
    invalid = TRANSITIONS.copy()
    invalid[0, 1] = [0.2, 0.2]

    with pytest.raises(ValueError, match="sum to 1"):
        value_iteration(REWARDS, invalid, gamma=0.5)