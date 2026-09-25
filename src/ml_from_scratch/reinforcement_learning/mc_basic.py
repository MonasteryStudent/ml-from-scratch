import numpy as np


def discounted_return(rewards, gamma):
    """Calculate the discounted return of an episode."""
    return_value = 0.0
    discount = 1.0

    for reward in rewards:
        return_value += discount * reward
        discount *= gamma

    return return_value


def generate_episode(initial_state, initial_action, policy, step):
    """Generate an episode beginning with a given state-action pair."""
    states = [initial_state]
    actions = []
    rewards = []

    state = initial_state
    action = initial_action

    while True:
        actions.append(action)

        next_state, reward, terminated = step(state, action)

        rewards.append(reward)
        states.append(next_state)

        if terminated:
            return states, actions, rewards

        state = next_state
        action = policy[state]


def estimate_action_value(
    state,
    action,
    policy,
    step,
    gamma,
    episodes_per_pair,
):
    """Estimate an action value by averaging sampled episode returns."""
    returns = []

    for _ in range(episodes_per_pair):
        _, _, rewards = generate_episode(
            state,
            action,
            policy,
            step,
        )

        return_value = discounted_return(rewards, gamma)
        returns.append(return_value)

    return np.mean(returns)


def estimate_action_values(
    policy,
    n_states,
    n_actions,
    step,
    gamma,
    episodes_per_pair,
):
    """Estimate the action values for all state-action pairs."""
    action_values = np.zeros((n_states, n_actions), dtype=float)

    for state in range(n_states):
        for action in range(n_actions):
            action_values[state, action] = estimate_action_value(
                state,
                action,
                policy,
                step,
                gamma,
                episodes_per_pair,
            )

    return action_values


def policy_improvement(action_values):
    """Return a greedy deterministic policy."""
    # np.argmax chooses the first maximizing action when there is a tie.
    improved_policy = np.argmax(action_values, axis=1)

    return improved_policy


def mc_basic(
    initial_policy,
    n_states,
    n_actions,
    step,
    gamma,
    episodes_per_pair,
):
    """Find a deterministic policy using Monte Carlo estimates."""
    policy = np.asarray(initial_policy, dtype=int).copy()

    while True:
        action_values = estimate_action_values(
            policy,
            n_states,
            n_actions,
            step,
            gamma,
            episodes_per_pair,
        )

        improved_policy = policy_improvement(action_values)

        if np.array_equal(improved_policy, policy):
            return action_values, policy

        policy = improved_policy