import numpy as np


def q_learning_update(
    current_action_value,
    reward,
    next_action_values,
    gamma,
    step_size,
):
    """Return the updated action value after one Q-learning step."""
    greatest_next_action_value = np.max(
        next_action_values
    )

    td_target = (
        reward
        + gamma * greatest_next_action_value
    )

    td_error = current_action_value - td_target

    updated_action_value = (
        current_action_value
        - step_size * td_error
    )

    return updated_action_value


def select_action(behavior_policy, state, random_generator):
    """Sample an action from the behavior policy of a state."""
    n_actions = behavior_policy.shape[1]

    action = random_generator.choice(
        n_actions,
        p=behavior_policy[state],
    )

    return int(action)


def target_policy_improvement(
    target_policy,
    action_values,
    state,
):
    """Return a greedy target policy for the visited state."""
    improved_target_policy = target_policy.copy()

    # np.argmax selects the first maximizing action in a tie.
    greedy_action = np.argmax(
        action_values[state]
    )

    improved_target_policy[state] = 0.0
    improved_target_policy[state, greedy_action] = 1.0

    return improved_target_policy


def generate_episode(
    initial_state,
    behavior_policy,
    step,
    random_generator,
):
    """Generate an episode by following the behavior policy."""
    states = [initial_state]
    actions = []
    rewards = []

    state = initial_state
    terminated = False

    while not terminated:
        action = select_action(
            behavior_policy,
            state,
            random_generator,
        )

        next_state, reward, terminated = step(
            state,
            action,
        )

        actions.append(action)
        rewards.append(reward)
        states.append(next_state)

        state = next_state

    return states, actions, rewards


def collect_episodes(
    initial_state,
    behavior_policy,
    step,
    n_episodes,
    random_generator,
):
    """Collect episodes by following the behavior policy."""
    episodes = []

    for _ in range(n_episodes):
        episode = generate_episode(
            initial_state=initial_state,
            behavior_policy=behavior_policy,
            step=step,
            random_generator=random_generator,
        )

        episodes.append(episode)

    return episodes


def q_learning(
    episodes,
    initial_action_values,
    gamma,
    step_size,
):
    """Learn optimal action values and a greedy target policy."""
    action_values = np.asarray(
        initial_action_values,
        dtype=float,
    ).copy()

    n_states, n_actions = action_values.shape

    target_policy = np.zeros(
        (n_states, n_actions),
        dtype=float,
    )

    for state in range(n_states):
        # np.argmax selects the first maximizing action in a tie.
        greedy_action = np.argmax(
            action_values[state]
        )
        target_policy[state, greedy_action] = 1.0

    for states, actions, rewards in episodes:
        for time_step, (state, action, reward) in enumerate(
            zip(states[:-1], actions, rewards)
        ):
            next_state = states[time_step + 1]

            if time_step == len(actions) - 1:
                next_action_values = np.zeros(
                    n_actions,
                    dtype=float,
                )
            else:
                next_action_values = action_values[
                    next_state
                ]

            action_values[state, action] = q_learning_update(
                current_action_value=action_values[state, action],
                reward=reward,
                next_action_values=next_action_values,
                gamma=gamma,
                step_size=step_size,
            )

            target_policy = target_policy_improvement(
                target_policy,
                action_values,
                state,
            )

    return action_values, target_policy