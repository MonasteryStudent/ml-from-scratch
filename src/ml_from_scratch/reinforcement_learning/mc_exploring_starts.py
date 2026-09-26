import numpy as np


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


def policy_improvement(policy, action_values, visited_states):
    """Return a greedy policy for the states visited in an episode."""
    improved_policy = policy.copy()

    for state in np.unique(visited_states):
        improved_policy[state] = np.argmax(action_values[state])

    return improved_policy


def generate_episode(
    initial_state,
    initial_action,
    policy,
    step,
    episode_length,
):
    """Generate an episode of a fixed length from a given starting pair."""
    states = [initial_state]
    actions = []
    rewards = []

    state = initial_state
    action = initial_action

    for time_step in range(episode_length):
        actions.append(action)

        next_state, reward = step(state, action)

        states.append(next_state)
        rewards.append(reward)

        if time_step < episode_length - 1:
            state = next_state
            action = policy[state]

    return states, actions, rewards


def select_starting_pair(
    n_states,
    n_actions,
    random_generator,
):
    """Select a state-action pair uniformly at random."""
    state = random_generator.integers(n_states)
    action = random_generator.integers(n_actions)

    return int(state), int(action)


def mc_exploring_starts(
    initial_policy,
    initial_action_values,
    step,
    gamma,
    n_episodes,
    episode_length,
    random_generator,
):
    """Estimate action values and improve a policy episode by episode."""
    policy = np.asarray(initial_policy, dtype=int).copy()
    action_values = np.asarray(
        initial_action_values,
        dtype=float,
    ).copy()

    n_states, n_actions = action_values.shape

    return_sums = np.zeros_like(action_values, dtype=float)
    visit_counts = np.zeros_like(action_values, dtype=int)

    for _ in range(n_episodes):
        initial_state, initial_action = select_starting_pair(
            n_states,
            n_actions,
            random_generator,
        )

        states, actions, rewards = generate_episode(
            initial_state,
            initial_action,
            policy,
            step,
            episode_length,
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
        )

    return action_values, policy