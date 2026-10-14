import numpy as np


def sarsa_update(
    current_action_value,
    reward,
    next_action_value,
    gamma,
    step_size,
):
    """Return the updated action value after one Sarsa step."""
    td_target = reward + gamma * next_action_value
    td_error = current_action_value - td_target

    updated_action_value = current_action_value - step_size * td_error

    return updated_action_value


def select_action(policy, state, random_generator):
    """Sample an action from the policy distribution of a state."""
    n_actions = policy.shape[1]

    action = random_generator.choice(
        n_actions,
        p=policy[state],
    )

    return int(action)


def policy_improvement(
    policy,
    action_values,
    state,
    epsilon,
):
    """Return an epsilon-greedy policy for the visited state."""
    improved_policy = policy.copy()
    n_actions = action_values.shape[1]

    greedy_action = np.argmax(
        action_values[state]
    )

    improved_policy[state] = epsilon / n_actions
    improved_policy[state, greedy_action] += 1 - epsilon

    return improved_policy


def sarsa(
    initial_policy,
    initial_action_values,
    initial_state,
    step,
    gamma,
    step_size,
    epsilon,
    n_episodes,
    random_generator,
):
    """Estimate action values and improve an epsilon-greedy policy."""
    policy = np.asarray(
        initial_policy,
        dtype=float,
    ).copy()

    action_values = np.asarray(
        initial_action_values,
        dtype=float,
    ).copy()

    for _ in range(n_episodes):
        state = initial_state

        action = select_action(
            policy,
            state,
            random_generator,
        )

        terminated = False

        while not terminated:
            next_state, reward, terminated = step(
                state,
                action,
            )

            if terminated:
                next_action_value = 0.0
            else:
                next_action = select_action(
                    policy,
                    next_state,
                    random_generator,
                )

                next_action_value = action_values[
                    next_state,
                    next_action,
                ]

            action_values[state, action] = sarsa_update(
                current_action_value=action_values[state, action],
                reward=reward,
                next_action_value=next_action_value,
                gamma=gamma,
                step_size=step_size,
            )

            policy = policy_improvement(
                policy,
                action_values,
                state,
                epsilon,
            )

            if not terminated:
                state = next_state
                action = next_action

    return action_values, policy