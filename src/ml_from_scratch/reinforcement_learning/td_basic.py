import numpy as np


def td_update(
    current_value,
    reward,
    next_state_value,
    gamma,
    step_size,
):
    """Return the updated state value after one TD step."""
    td_target = reward + gamma * next_state_value
    td_error = current_value - td_target

    updated_value = current_value - step_size * td_error

    return updated_value


def td_state_values(
    initial_values,
    policy,
    initial_state,
    step,
    gamma,
    step_size,
    n_steps,
):
    """Estimate state values of a fixed deterministic policy."""
    values = np.asarray(initial_values, dtype=float).copy()
    policy = np.asarray(policy, dtype=int)

    state = initial_state
    value_history = [values.copy()]

    for _ in range(n_steps):
        action = policy[state]
        next_state, reward = step(state, action)

        values[state] = td_update(
            current_value=values[state],
            reward=reward,
            next_state_value=values[next_state],
            gamma=gamma,
            step_size=step_size,
        )

        value_history.append(values.copy())
        state = next_state

    return values, np.asarray(value_history)