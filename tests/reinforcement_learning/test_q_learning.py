import numpy as np

from ml_from_scratch.reinforcement_learning.q_learning import (
    q_learning_update,
    select_action,
    target_policy_improvement,
    generate_episode,
    collect_episodes,
    q_learning,
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


def test_q_learning_update_uses_greatest_next_action_value():
    next_action_values = np.array([
        3.0,
        6.0,
    ])

    updated_action_value = q_learning_update(
        current_action_value=2.0,
        reward=1.0,
        next_action_values=next_action_values,
        gamma=0.5,
        step_size=0.25,
    )

    assert updated_action_value == 2.5


def test_select_action_follows_behavior_policy():
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


def test_target_policy_improvement_updates_visited_state_greedily():
    target_policy = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])

    action_values = np.array([
        [1.0, 3.0],
        [4.0, 2.0],
        [5.0, 0.0],
    ])

    improved_target_policy = target_policy_improvement(
        target_policy,
        action_values,
        state=0,
    )

    expected_target_policy = np.array([
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])

    np.testing.assert_array_equal(
        improved_target_policy,
        expected_target_policy,
    )

    np.testing.assert_array_equal(
        target_policy,
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
    )


def test_generate_episode_follows_behavior_policy():
    behavior_policy = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    states, actions, rewards = generate_episode(
        initial_state=0,
        behavior_policy=behavior_policy,
        step=deterministic_step,
        random_generator=np.random.default_rng(0),
    )

    assert states == [0, 1, 2]
    assert actions == [1, 0]
    assert rewards == [1.0, 2.0]


def test_collect_episodes_creates_offline_dataset():
    behavior_policy = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    episodes = collect_episodes(
        initial_state=0,
        behavior_policy=behavior_policy,
        step=deterministic_step,
        n_episodes=3,
        random_generator=np.random.default_rng(0),
    )

    assert len(episodes) == 3

    for states, actions, rewards in episodes:
        assert states == [0, 1, 2]
        assert actions == [1, 0]
        assert rewards == [1.0, 2.0]


def test_q_learning_processes_collected_episode():
    episodes = [
        (
            [0, 1, 2],
            [1, 0],
            [1.0, 2.0],
        ),
    ]

    initial_action_values = np.zeros((2, 2))

    action_values, target_policy = q_learning(
        episodes=episodes,
        initial_action_values=initial_action_values,
        gamma=0.5,
        step_size=0.5,
    )

    expected_action_values = np.array([
        [0.0, 0.5],
        [1.0, 0.0],
    ])

    expected_target_policy = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    np.testing.assert_allclose(
        action_values,
        expected_action_values,
    )

    np.testing.assert_array_equal(
        target_policy,
        expected_target_policy,
    )

    np.testing.assert_array_equal(
        initial_action_values,
        np.zeros((2, 2)),
    )


def test_offline_q_learning_learns_greedy_target_policy():
    behavior_policy = np.full(
        (2, 2),
        0.5,
    )

    initial_action_values = np.zeros((2, 2))

    episodes = collect_episodes(
        initial_state=0,
        behavior_policy=behavior_policy,
        step=deterministic_step,
        n_episodes=200,
        random_generator=np.random.default_rng(0),
    )

    action_values, target_policy = q_learning(
        episodes=episodes,
        initial_action_values=initial_action_values,
        gamma=0.5,
        step_size=0.1,
    )

    expected_target_policy = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    np.testing.assert_array_equal(
        target_policy,
        expected_target_policy,
    )

    assert action_values[0, 1] > action_values[0, 0]
    assert action_values[1, 0] > action_values[1, 1]

    np.testing.assert_array_equal(
        behavior_policy,
        np.full((2, 2), 0.5),
    )

    np.testing.assert_array_equal(
        initial_action_values,
        np.zeros((2, 2)),
    )