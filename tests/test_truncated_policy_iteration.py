import numpy as np

from ml_from_scratch.reinforcement_learning.truncated_policy_iteration import (
    policy_improvement,
    truncated_policy_evaluation,
    truncated_policy_iteration,
)

# Expected immediate rewards r(s, a).
# Rows: states s0 and s1.
# Columns: actions a0=left, a1=stay, and a2=right.
EXPECTED_REWARDS = np.array([
    [-1.0, 0.0,  1.0],  # Rewards in s0 for a0, a1, and a2
    [ 0.0, 1.0, -1.0],  # Rewards in s1 for a0, a1, and a2
])

# Transition probabilities p(s' | s, a).
# Each innermost vector gives the probabilities of reaching [s0, s1].
TRANSITIONS = np.array([
    [  # Current state s0
        [1.0, 0.0],  # a0 leads to s0
        [1.0, 0.0],  # a1 leads to s0
        [0.0, 1.0],  # a2 leads to s1
    ],
    [  # Current state s1
        [1.0, 0.0],  # a0 leads to s0
        [0.0, 1.0],  # a1 leads to s1
        [0.0, 1.0],  # a2 leads to s1
    ],
])


def test_truncated_policy_evaluation_performs_fixed_number_of_updates():
    policy = np.array([0, 0])
    initial_values = np.zeros(2)

    values = truncated_policy_evaluation(
        policy,
        initial_values,
        EXPECTED_REWARDS,
        TRANSITIONS,
        gamma=0.9,
        evaluation_steps=3
    )

    np.testing.assert_allclose(values, [-2.71, -1.71])


def test_policy_improvement_selects_greedy_actions():
    values = np.array([-10.0, -9.0])

    q_values, policy = policy_improvement(
        values,
        EXPECTED_REWARDS,
        TRANSITIONS,
        gamma=0.9,
    )

    expected_q_values = np.array([
        [-10.0, -9.0, -7.1],
        [ -9.0, -7.1, -9.1],
    ])

    np.testing.assert_allclose(q_values, expected_q_values)
    np.testing.assert_array_equal(policy, [2, 1])


def test_truncated_policy_iteration_finds_optimal_policy():
    initial_policy = np.array([0, 0])

    values, policy = truncated_policy_iteration(
        initial_policy,
        EXPECTED_REWARDS,
        TRANSITIONS,
        gamma=0.9,
        evaluation_steps=3,
    )

    np.testing.assert_allclose(values, [10.0, 10.0])
    np.testing.assert_array_equal(policy, [2, 1])


def test_truncated_policy_iteration_converges_with_one_evaluation_step():
    initial_policy = np.array([0, 0])

    values, policy = truncated_policy_iteration(
        initial_policy,
        EXPECTED_REWARDS,
        TRANSITIONS,
        gamma=0.9,
        evaluation_steps=1,
    )

    np.testing.assert_allclose(values, [10.0, 10.0])
    np.testing.assert_array_equal(policy, [2, 1])