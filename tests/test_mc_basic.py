import numpy as np

from ml_from_scratch.reinforcement_learning.mc_basic import (
    discounted_return,
    generate_episode,
    estimate_action_value,
    estimate_action_values,
    policy_improvement,
    mc_basic,
)

REWARD_TABLE = np.array([
    [0.0, 1.0],
    [2.0, 3.0],
])

NEXT_STATES = np.array([
    [2, 1],
    [2, 2],
])

TERMINATED = np.array([
    [True, False],
    [True, True],
])


def deterministic_step(state, action):
    """Simulate one transition in the deterministic test environment."""
    next_state = NEXT_STATES[state, action]
    reward = REWARD_TABLE[state, action]
    terminated = TERMINATED[state, action]

    return next_state, reward, terminated


def test_discounted_return_matches_manual_calculation():
    rewards = [1.0, 2.0, 3.0]

    return_value = discounted_return(rewards, gamma=0.5)

    assert return_value == 2.75


def test_generate_episode_starts_with_given_state_action_pair():
    policy = np.array([0, 0])

    states, actions, rewards = generate_episode(
        initial_state=0,
        initial_action=1,
        policy=policy,
        step=deterministic_step,
    )

    assert states == [0, 1, 2]
    assert actions == [1, 0]
    assert rewards == [1.0, 2.0]


def test_estimate_action_value_averages_episode_returns():
    sampled_rewards = iter([1.0, 3.0])
    policy = np.array([0])

    def sampled_step(state, action):
        reward = next(sampled_rewards)
        return 1, reward, True

    action_value = estimate_action_value(
        state=0,
        action=0,
        policy=policy,
        step=sampled_step,
        gamma=0.5,
        episodes_per_pair=2,
    )

    assert action_value == 2.0


def test_estimate_action_values_for_all_state_action_pairs():
    policy = np.array([0, 0])

    action_values = estimate_action_values(
        policy=policy,
        n_states=2,
        n_actions=2,
        step=deterministic_step,
        gamma=0.9,
        episodes_per_pair=1,
    )

    expected_action_values = np.array([
        [0.0, 2.8],
        [2.0, 3.0],
    ])

    np.testing.assert_allclose(
        action_values,
        expected_action_values,
    )


def test_policy_improvement_selects_greedy_actions():
    action_values = np.array([
        [0.0, 2.8],
        [2.0, 3.0],
    ])

    improved_policy = policy_improvement(action_values)

    np.testing.assert_array_equal(
        improved_policy,
        [1, 1],
    )


def test_mc_basic_finds_optimal_policy():
    initial_policy = np.array([0, 0])

    action_values, policy = mc_basic(
        initial_policy=initial_policy,
        n_states=2,
        n_actions=2,
        step=deterministic_step,
        gamma=0.9,
        episodes_per_pair=1,
    )

    expected_action_values = np.array([
        [0.0, 3.7],
        [2.0, 3.0],
    ])

    np.testing.assert_allclose(
        action_values,
        expected_action_values,
    )
    np.testing.assert_array_equal(policy, [1, 1])
    np.testing.assert_array_equal(initial_policy, [0, 0])