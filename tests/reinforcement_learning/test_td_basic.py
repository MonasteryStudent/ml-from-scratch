import numpy as np

from ml_from_scratch.reinforcement_learning.td_basic import (
    td_update,
    td_state_values,
)

NEXT_STATE_TABLE = np.array([
    [1],
    [0],
])

REWARD_TABLE = np.array([
    [1.0],
    [0.0],
])


def deterministic_step(state, action):
    """Simulate one transition in the deterministic test environment."""
    next_state = NEXT_STATE_TABLE[state, action]
    reward = REWARD_TABLE[state, action]

    return next_state, reward


def test_td_update_moves_value_toward_target():
    updated_value = td_update(
        current_value=2.0,
        reward=1.0,
        next_state_value=6.0,
        gamma=0.5,
        step_size=0.25,
    )

    assert updated_value == 2.5


def test_td_state_values_updates_after_every_transition():
    values, value_history = td_state_values(
        initial_values=[0.0, 0.0],
        policy=[0, 0],
        initial_state=0,
        step=deterministic_step,
        gamma=0.5,
        step_size=0.5,
        n_steps=2,
    )

    np.testing.assert_allclose(
        value_history,
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.5, 0.125],
        ],
    )

    np.testing.assert_allclose(
        values,
        [0.5, 0.125],
    )


def test_td_state_values_converges_to_true_values():
    values, _ = td_state_values(
        initial_values=[0.0, 0.0],
        policy=[0, 0],
        initial_state=0,
        step=deterministic_step,
        gamma=0.5,
        step_size=0.1,
        n_steps=400,
    )

    np.testing.assert_allclose(
        values,
        [4 / 3, 2 / 3],
        atol=1e-4,
    )


def test_td_state_values_does_not_modify_initial_values():
    initial_values = np.array([0.0, 0.0])

    td_state_values(
        initial_values=initial_values,
        policy=[0, 0],
        initial_state=0,
        step=deterministic_step,
        gamma=0.5,
        step_size=0.5,
        n_steps=2,
    )

    np.testing.assert_array_equal(
        initial_values,
        [0.0, 0.0],
    )