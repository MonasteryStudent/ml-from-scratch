import numpy as np


def policy_evaluation(policy, expected_rewards, transitions, gamma):
    """Calculate the state values of a fixed deterministic policy."""
    n_states = expected_rewards.shape[0]
    states = np.arange(n_states)

    policy_rewards = expected_rewards[states, policy]
    policy_transitions = transitions[states, policy]

    identity = np.eye(n_states)
    values = np.linalg.solve(
        identity - gamma * policy_transitions,
        policy_rewards
    )

    return values


def policy_improvement(values, expected_rewards, transitions, gamma):
    """Return action values and a greedy deterministic policy."""
    q_values = expected_rewards + gamma * (transitions @ values)
    improved_policy = np.argmax(q_values, axis=1)

    return q_values, improved_policy


def policy_iteration(initial_policy, expected_rewards, transitions, gamma):
    """Find optimal state values and a deterministic policy."""
    policy = np.asarray(initial_policy, dtype=int).copy()

    while True:
        values = policy_evaluation(
            policy, expected_rewards, transitions, gamma
        )

        _, improved_policy = policy_improvement(
            values, expected_rewards, transitions, gamma
        )

        if np.array_equal(improved_policy, policy):
            return values, policy

        policy = improved_policy