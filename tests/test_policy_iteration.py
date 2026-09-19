import numpy as np

from ml_from_scratch.reinforcement_learning.policy_iteration import (
    policy_evaluation,
    policy_improvement,
    policy_iteration,
)

# Expected immediate rewards r(s, a).
# Actions: a0:=left, a1:=stay, a2:=right.
EXPECTED_REWARDS = np.array([
    [-1.0, 0.0,  1.0], # Rewards in s0 for a0, a1, and a2
    [ 0.0, 1.0, -1.0], # Rewards in s1 for a0, a1, and a2
])

TRANSITIONS = np.array([
    [   # current state s0
        [1.0, 0.0], # a0 leads to s0
        [1.0, 0.0], # a1 leads to s0
        [0.0, 1.0], # a2 leads to s1
    ],  
    [   # current state s1
        [1.0, 0.0], # a0 leads to s0
        [0.0, 1.0], # a1 leads to s1
        [0.0, 1.0], # a2 leads to s1
    ],  
])


def test_policy_evaluation_for_initial_policy():
    # Select left in both states.
    policy = np.array([0, 0])

    values = policy_evaluation(
        policy,
        EXPECTED_REWARDS,
        TRANSITIONS,
        gamma=0.9
    )

    np.testing.assert_allclose(values, [-10.0, -9.0])


def test_policy_improvement_for_initial_policy():
    values = np.array([-10.0, -9.0])

    q_values, improved_policy = policy_improvement(
        values,
        EXPECTED_REWARDS,
        TRANSITIONS,
        gamma=0.9
    )

    np.testing.assert_allclose(q_values, [
        [-10.0, -9.0, -7.1],
        [ -9.0, -7.1, -9.1],
    ])
    # The policy [2, 1] means right in s0 and stay in s1.
    np.testing.assert_array_equal(improved_policy, [2, 1])


def test_policy_iteration_finds_optimal_policy():
    initial_policy = np.array([0, 0])

    values, policy = policy_iteration(
        initial_policy,
        EXPECTED_REWARDS,
        TRANSITIONS,
        gamma=0.9
    )

    np.testing.assert_allclose(values, [10.0, 10.0])
    np.testing.assert_array_equal(policy, [2, 1])
    np.testing.assert_array_equal(initial_policy, [0, 0])