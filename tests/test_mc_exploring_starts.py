import numpy as np

from ml_from_scratch.reinforcement_learning.mc_exploring_starts import (
    discounted_returns,
    update_action_values,
    policy_improvement,
    generate_episode,
    select_starting_pair,
    mc_exploring_starts,
)

REWARD_TABLE = np.array([
    [0.0, 1.0],
    [2.0, 3.0],
])

NEXT_STATE_TABLE = np.array([
    [0, 1],
    [0, 1],
])


def deterministic_step(state, action):
    """Simulate one transition in the deterministic test environment."""
    next_state = NEXT_STATE_TABLE[state, action]
    reward = REWARD_TABLE[state, action]

    return next_state, reward


def test_discounted_returns_for_every_step():
    rewards = [1.0, 2.0, 3.0]

    returns = discounted_returns(rewards, gamma=0.5)

    np.testing.assert_allclose(
        returns,
        [2.75, 3.5, 3.0],
    )


def test_update_action_values_uses_every_visit():
    states = [0, 1, 0, 1]
    actions = [0, 1, 0]
    rewards = [1.0, 2.0, 3.0]

    return_sums = np.zeros((2, 2), dtype=float)
    visit_counts = np.zeros((2, 2), dtype=int)
    action_values = np.zeros((2, 2), dtype=float)

    update_action_values(
        states,
        actions,
        rewards,
        return_sums,
        visit_counts,
        action_values,
        gamma=0.5,
    )

    expected_return_sums = np.array([
        [5.75, 0.0],
        [0.0, 3.5],
    ])

    expected_visit_counts = np.array([
        [2, 0],
        [0, 1],
    ])

    expected_action_values = np.array([
        [2.875, 0.0],
        [0.0, 3.5],
    ])

    np.testing.assert_allclose(
        return_sums,
        expected_return_sums,
    )
    np.testing.assert_array_equal(
        visit_counts,
        expected_visit_counts,
    )
    np.testing.assert_allclose(
        action_values,
        expected_action_values,
    )


def test_policy_improvement_updates_only_visited_states():
    policy = np.array([0, 1, 1])

    action_values = np.array([
        [1.0, 3.0],
        [4.0, 2.0],
        [5.0, 0.0],
    ])

    improved_policy = policy_improvement(
        policy,
        action_values,
        visited_states=[0, 1, 0],
    )

    np.testing.assert_array_equal(
        improved_policy,
        [1, 0, 1],
    )
    np.testing.assert_array_equal(
        policy,
        [0, 1, 1],
    )


def test_generate_episode_uses_starting_action_then_policy():
    policy = np.array([0, 0])

    states, actions, rewards = generate_episode(
        initial_state=0,
        initial_action=1,
        policy=policy,
        step=deterministic_step,
        episode_length=3,
    )

    assert states == [0, 1, 0, 0]
    assert actions == [1, 0, 0]
    assert rewards == [1.0, 2.0, 0.0]


def test_select_starting_pair_can_select_every_pair():
    random_generator = np.random.default_rng(0)

    selected_pairs = {
        select_starting_pair(
            n_states=2,
            n_actions=2,
            random_generator=random_generator,
        )
        for _ in range(100)
    }

    assert selected_pairs == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }


def test_mc_exploring_starts_finds_greedy_policy():
    initial_policy = np.array([0, 0])
    initial_action_values = np.zeros((2, 2))

    action_values, policy = mc_exploring_starts(
        initial_policy=initial_policy,
        initial_action_values=initial_action_values,
        step=deterministic_step,
        gamma=0.5,
        n_episodes=200,
        episode_length=3,
        random_generator=np.random.default_rng(0),
    )

    np.testing.assert_array_equal(policy, [1, 1])

    assert action_values[0, 1] > action_values[0, 0]
    assert action_values[1, 1] > action_values[1, 0]

    np.testing.assert_array_equal(
        initial_policy,
        [0, 0],
    )
    np.testing.assert_array_equal(
        initial_action_values,
        np.zeros((2, 2)),
    )