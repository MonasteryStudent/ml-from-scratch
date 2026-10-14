import numpy as np

from ml_from_scratch.reinforcement_learning.sarsa import (
    sarsa_update,
    select_action,
    policy_improvement,
    sarsa,
)

NEXT_STATE_TABLE = np.array([
    [2, 1],
    [2, 2],
])

REWARD_TABLE = np.array([
    [0.0, 1.0],
    [2.0, 0.0],
])

TERMINATED_TABLE = np.array([
    [True, False],
    [True, True],
])


def deterministic_step(state, action):
    """Simulate one transition in the deterministic test environment."""
    next_state = NEXT_STATE_TABLE[state, action]
    reward = REWARD_TABLE[state, action]
    terminated = TERMINATED_TABLE[state, action]

    return next_state, reward, terminated


def test_sarsa_update_moves_action_value_toward_target():
    updated_action_value = sarsa_update(
        current_action_value=2.0,
        reward=1.0,
        next_action_value=6.0,
        gamma=0.5,
        step_size=0.25,
    )

    assert updated_action_value == 2.5


def test_select_action_follows_policy():
    policy = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    random_generator = np.random.default_rng(0)

    action_in_state_0 = select_action(
        policy,
        state=0,
        random_generator=random_generator,
    )

    action_in_state_1 = select_action(
        policy,
        state=1,
        random_generator=random_generator,
    )

    assert action_in_state_0 == 1
    assert action_in_state_1 == 0


def test_policy_improvement_updates_visited_state_epsilon_greedily():
    policy = np.full((3, 2), 0.5)

    action_values = np.array([
        [1.0, 3.0],
        [4.0, 2.0],
        [5.0, 0.0],
    ])

    improved_policy = policy_improvement(
        policy,
        action_values,
        state=0,
        epsilon=0.2,
    )

    expected_policy = np.array([
        [0.1, 0.9],
        [0.5, 0.5],
        [0.5, 0.5],
    ])

    np.testing.assert_allclose(
        improved_policy,
        expected_policy,
    )

    np.testing.assert_allclose(
        policy,
        np.full((3, 2), 0.5),
    )


def test_sarsa_updates_action_values_during_episode():
    initial_policy = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    initial_action_values = np.zeros((2, 2))

    action_values, policy = sarsa(
        initial_policy=initial_policy,
        initial_action_values=initial_action_values,
        initial_state=0,
        step=deterministic_step,
        gamma=0.5,
        step_size=0.5,
        epsilon=0.0,
        n_episodes=1,
        random_generator=np.random.default_rng(0),
    )

    np.testing.assert_allclose(
        action_values,
        [
            [0.0, 0.5],
            [1.0, 0.0],
        ],
    )

    np.testing.assert_allclose(
        policy,
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
    )


def test_sarsa_learns_best_epsilon_greedy_policy():
    initial_policy = np.full((2, 2), 0.5)
    initial_action_values = np.zeros((2, 2))

    action_values, policy = sarsa(
        initial_policy=initial_policy,
        initial_action_values=initial_action_values,
        initial_state=0,
        step=deterministic_step,
        gamma=0.5,
        step_size=0.1,
        epsilon=0.2,
        n_episodes=100,
        random_generator=np.random.default_rng(0),
    )

    expected_policy = np.array([
        [0.1, 0.9],
        [0.9, 0.1],
    ])

    np.testing.assert_allclose(
        policy,
        expected_policy,
    )

    assert action_values[0, 1] > action_values[0, 0]
    assert action_values[1, 0] > action_values[1, 1]

    np.testing.assert_allclose(
        initial_policy,
        np.full((2, 2), 0.5),
    )

    np.testing.assert_array_equal(
        initial_action_values,
        np.zeros((2, 2)),
    )