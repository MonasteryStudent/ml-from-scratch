import numpy as np


def truncated_policy_evaluation(
    policy,
    values,
    expected_rewards,
    transitions,
    gamma,
    evaluation_steps,
):
    """Approximate policy values using a fixed number of updates."""
    n_states = expected_rewards.shape[0]
    states = np.arange(n_states)

    policy_rewards = expected_rewards[states, policy]
    policy_transitions = transitions[states, policy]

    updated_values = np.asarray(values, dtype=float).copy()

    for _ in range(evaluation_steps):
        updated_values = (
            policy_rewards
            + gamma * (policy_transitions @ updated_values)
        )

    return updated_values


def policy_improvement(values, expected_rewards, transitions, gamma):
    """Return action values and a greedy policy."""
    q_values = (expected_rewards + gamma * (transitions @ values))

    # np.argmax chooses the first maximizing action when there is a tie.
    improved_policy = np.argmax(q_values, axis=1)

    return q_values, improved_policy


def truncated_policy_iteration(
    initial_policy,
    expected_rewards,
    transitions,
    gamma,
    evaluation_steps,
    tolerance=1e-8,
):
    """Find approximate optimal values and a deterministic policy."""
    policy = np.asarray(initial_policy, dtype=int).copy()
    values = np.zeros(expected_rewards.shape[0], dtype=float)

    while True:
        previous_values = values.copy()

        values = truncated_policy_evaluation(
            policy,
            values,
            expected_rewards,
            transitions,
            gamma,
            evaluation_steps,
        )

        _, improved_policy = policy_improvement(
            values,
            expected_rewards,
            transitions,
            gamma,
        )

        difference = np.max(np.abs(values - previous_values))

        if (
            difference <= tolerance
            and np.array_equal(improved_policy, policy)
        ):
            return values, policy

        policy = improved_policy