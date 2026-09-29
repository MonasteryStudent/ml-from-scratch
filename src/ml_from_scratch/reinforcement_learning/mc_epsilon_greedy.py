import numpy as np


def policy_improvement(
    policy,
    action_values,
    visited_states,
    epsilon,
):
    """Return an epsilon-greedy policy for the visited states."""
    improved_policy = policy.copy()
    n_actions = action_values.shape[1]

    for state in np.unique(visited_states):
        greedy_action = np.argmax(action_values[state])

        improved_policy[state] = epsilon / n_actions
        improved_policy[state, greedy_action] += 1 - epsilon

    return improved_policy


def select_action(policy, state, random_generator):
    """Sample an action from the policy distribution of a state."""
    n_actions = policy.shape[1]

    action = random_generator.choice(
        n_actions,
        p=policy[state],
    )

    return int(action)


def generate_episode(
    initial_state,
    policy,
    step,
    episode_length,
    random_generator,
):
    """Generate an episode by sampling actions from a stochastic policy."""
    states = [initial_state]
    actions = []
    rewards = []

    state = initial_state

    for _ in range(episode_length):
        action = select_action(
            policy,
            state,
            random_generator,
        )

        next_state, reward = step(state, action)

        actions.append(action)
        rewards.append(reward)
        states.append(next_state)

        state = next_state

    return states, actions, rewards


def discounted_returns(rewards, gamma):
    """Calculate the discounted return from every step of an episode."""
    returns = np.zeros(len(rewards), dtype=float)
    return_value = 0.0

    for time_step in reversed(range(len(rewards))):
        return_value = rewards[time_step] + gamma * return_value
        returns[time_step] = return_value

    return returns


def update_action_values(
    states,
    actions,
    rewards,
    return_sums,
    visit_counts,
    action_values,
    gamma,
):
    """Update return statistics and action values in place."""
    returns = discounted_returns(rewards, gamma)

    for state, action, return_value in zip(
        states[:-1],
        actions,
        returns,
    ):
        return_sums[state, action] += return_value
        visit_counts[state, action] += 1

        action_values[state, action] = (
            return_sums[state, action]
            / visit_counts[state, action]
        )


def mc_epsilon_greedy(
    initial_policy,
    initial_action_values,
    initial_state,
    step,
    gamma,
    epsilon,
    n_episodes,
    episode_length,
    random_generator,
):
    """Estimate action values and improve an epsilon-greedy policy."""
    policy = np.asarray(initial_policy, dtype=float).copy()
    action_values = np.asarray(
        initial_action_values,
        dtype=float,
    ).copy()

    return_sums = np.zeros_like(action_values, dtype=float)
    visit_counts = np.zeros_like(action_values, dtype=int)

    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(
            initial_state,
            policy,
            step,
            episode_length,
            random_generator,
        )

        update_action_values(
            states,
            actions,
            rewards,
            return_sums,
            visit_counts,
            action_values,
            gamma,
        )

        policy = policy_improvement(
            policy,
            action_values,
            states[:-1],
            epsilon,
        )

    return action_values, policy