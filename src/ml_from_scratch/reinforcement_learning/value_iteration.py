import numpy as np


def bellman_optimality_update(values, expected_rewards, transitions, gamma):
    """Compute q-values, greedy actions, and new values from the OLD values."""
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


def bellman_optimality_update_vectorized(
    values,
    expected_rewards,
    transitions,
    gamma
):
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


def _validate_value_iteration_inputs(
    expected_rewards, transitions, gamma, tolerance, max_iterations
):
    """Validate the model and value-iteration parameters."""
    if expected_rewards.ndim != 2:
        raise ValueError(
            "expected_rewards must have shape (states, actions)"
        )

    n_states, n_actions = expected_rewards.shape

    if n_states == 0 or n_actions == 0:
        raise ValueError("at least one state and action are required")

    if transitions.shape != (n_states, n_actions, n_states):
        raise ValueError(
            "transitions must have shape (states, actions, states)"
        )

    if not np.all(np.isfinite(expected_rewards)):
        raise ValueError("expected_rewards must contain finite numbers")

    if not np.all(np.isfinite(transitions)):
        raise ValueError("transitions must contain finite numbers")

    if np.any(transitions < 0):
        raise ValueError("transition probabilities must be nonnegative")

    if not np.allclose(transitions.sum(axis=2), 1):
        raise ValueError(
            "each transition distribution must sum to 1"
        )

    if not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be in [0, 1)")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(
            "tolerance must be a positive finite number"
        )

    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError(
            "max_iterations must be a positive integer"
        )


def value_iteration(
    expected_rewards, 
    transitions,
    gamma,
    tolerance=1e-8,
    max_iterations=10000,
    initial_values=None,
):
    """Return approximate optimal values, greedy actions, and value history.

    expected_rewards[state, action] is the expected immediate reward.
    transitions[state, action, next_state] is the transition probability.
    Each action is assumed available in every state.
    """
    expected_rewards = np.asarray(expected_rewards, dtype=float)
    transitions = np.asarray(transitions, dtype=float)

    _validate_value_iteration_inputs(
        expected_rewards, transitions, gamma, tolerance, max_iterations
    )

    n_states = expected_rewards.shape[0]

    if initial_values is None:
        values = np.zeros(n_states, dtype=float)
    else:
        values = np.asarray(initial_values, dtype=float).copy()
        if values.shape != (n_states,) or not np.all(np.isfinite(values)):
            raise ValueError("initial_values must be a finite value for every state")

    value_history = [values.copy()]

    for _ in range(max_iterations):
        _, greedy_actions, new_values = bellman_optimality_update_vectorized(
            values, expected_rewards, transitions, gamma
        )
        difference = np.max(np.abs(new_values - values))
        value_history.append(new_values.copy())
        values = new_values

        if difference <= tolerance:
            return values, greedy_actions, np.asarray(value_history)

    raise RuntimeError("value iteration did not converge within max_iterations")