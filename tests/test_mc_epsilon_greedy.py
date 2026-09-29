import numpy as np

from ml_from_scratch.reinforcement_learning.mc_epsilon_greedy import (
    policy_improvement,
    select_action,
    generate_episode,
    discounted_returns,
    update_action_values,
    mc_epsilon_greedy,
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


def test_policy_improvement_updates_visited_states_epsilon_greedily():
    policy = np.full((3, 2), 0.5)

    action_values = np.array([
        [1.0, 3.0],
        [4.0, 2.0],
        [5.0, 0.0],
    ])

    improved_policy = policy_improvement(
        policy,
        action_values,
        visited_states=[0, 1, 0],
        epsilon=0.2,
    )

    expected_policy = np.array([
        [0.1, 0.9],
        [0.9, 0.1],
        [0.5, 0.5],
    ])

    np.testing.assert_allclose(
        improved_policy,
        expected_policy,
    )
    np.testing.assert_allclose(
        improved_policy.sum(axis=1),
        [1.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(
        policy,
        np.full((3, 2), 0.5),
    )


def test_select_action_follows_policy_distribution():
    policy = np.array([
        [0.1, 0.9],
    ])
    random_generator = np.random.default_rng(0)

    selected_actions = [
        select_action(
            policy,
            state=0,
            random_generator=random_generator,
        )
        for _ in range(10_000)
    ]

    frequencies = np.bincount(
        selected_actions,
        minlength=2,
    ) / len(selected_actions)

    np.testing.assert_allclose(
        frequencies,
        [0.1, 0.9],
        atol=0.02,
    )


def test_generate_episode_selects_actions_from_policy():
    policy = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    states, actions, rewards = generate_episode(
        initial_state=0,
        policy=policy,
        step=deterministic_step,
        episode_length=3,
        random_generator=np.random.default_rng(0),
    )

    assert states == [0, 1, 0, 1]
    assert actions == [1, 0, 1]
    assert rewards == [1.0, 2.0, 1.0]


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

    np.testing.assert_allclose(
        return_sums,
        [
            [5.75, 0.0],
            [0.0, 3.5],
        ],
    )
    np.testing.assert_array_equal(
        visit_counts,
        [
            [2, 0],
            [0, 1],
        ],
    )
    np.testing.assert_allclose(
        action_values,
        [
            [2.875, 0.0],
            [0.0, 3.5],
        ],
    )


def test_mc_epsilon_greedy_finds_best_soft_policy():
    initial_policy = np.full((2, 2), 0.5)
    initial_action_values = np.zeros((2, 2))

    action_values, policy = mc_epsilon_greedy(
        initial_policy=initial_policy,
        initial_action_values=initial_action_values,
        initial_state=0,
        step=deterministic_step,
        gamma=0.5,
        epsilon=0.2,
        n_episodes=1000,
        episode_length=3,
        random_generator=np.random.default_rng(0),
    )

    expected_policy = np.array([
        [0.1, 0.9],
        [0.1, 0.9],
    ])

    np.testing.assert_allclose(
        policy,
        expected_policy,
    )

    assert action_values[0, 1] > action_values[0, 0]
    assert action_values[1, 1] > action_values[1, 0]

    np.testing.assert_allclose(
        initial_policy,
        np.full((2, 2), 0.5),
    )
    np.testing.assert_array_equal(
        initial_action_values,
        np.zeros((2, 2)),
    )