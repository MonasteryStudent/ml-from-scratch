import numpy as np


def bellman_optimality_update(values, expected_rewards, transitions, gamma):
    """Compute q-values, greedy actions, and new values from the old values."""
    n_states, n_actions = expected_rewards.shape
    q_values = np.zeros((n_states, n_actions), dtype=float)

    for state in range(n_states):
        for action in range(n_actions):
            future_value = 0.0
            for next_state in range(n_states):
                future_value += (
                    transitions[state, action, next_state] * values[next_state]
                )

            q_values[state, action] = (
                expected_rewards[state, action] + gamma * future_value
            )

    # np.argmax chooses the first maximizing action when there is a tie.
    greedy_actions = np.argmax(q_values, axis=1)
    new_values = np.max(q_values, axis=1)

    return q_values, greedy_actions, new_values


def bellman_optimality_update_vectorized(values, expected_rewards, transitions, gamma):
    """Compute q-values, greedy actions, and new values using vectorization."""
    expected_future_values = transitions @ values

    q_values = (
        expected_rewards
        + gamma * expected_future_values
    )

    # np.argmax chooses the first maximizing action when there is a tie.
    greedy_actions = np.argmax(q_values, axis=1)
    new_values = np.max(q_values, axis=1)

    return q_values, greedy_actions, new_values


def value_iteration(
    expected_rewards, 
    transitions, 
    gamma, 
    tolerance=1e-8, 
    initial_values=None,
):
    """Return approximate optimal values, greedy actions, and value history."""
    expected_rewards = np.asarray(expected_rewards, dtype=float)
    transitions = np.asarray(transitions, dtype=float)

    n_states = expected_rewards.shape[0]

    if initial_values is None:
        values = np.zeros(n_states, dtype=float)
    else:
        values = np.asarray(initial_values, dtype=float).copy()

    value_history = [values.copy()]

    while True:
        _, greedy_actions, new_values = bellman_optimality_update_vectorized(
            values, expected_rewards, transitions, gamma
        )

        difference = np.max(np.abs(new_values - values))

        value_history.append(new_values.copy())
        values = new_values

        if difference <= tolerance:
            return values, greedy_actions, np.asarray(value_history)