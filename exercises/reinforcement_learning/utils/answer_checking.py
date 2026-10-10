import numpy as np

from IPython.display import Markdown, display


TD_BASIC_EXPECTED_ANSWERS = np.array([
    [0.0,     0.0,     1.0,      -1.0,      0.5],
    [0.0,     0.5,     0.25,     -0.25,     0.125],
    [0.5,     0.125,   1.0625,   -0.5625,   0.78125],
    [0.125,   0.78125, 0.390625, -0.265625, 0.2578125],
])

TD_BASIC_HEADERS = [
    r"$t$",
    r"$s_t$",
    r"$r_{t+1}$",
    r"$s_{t+1}$",
    r"$v_t(s_t)$",
    r"$v_t(s_{t+1})$",
    r"$\bar{v}_t$",
    r"$\delta_t$",
    r"$v_{t+1}(s_t)$",
]


def display_table(answer_rows, headers):
    """Display the submitted answers as a Markdown table."""
    table = "| " + " | ".join(headers) + " |\n"
    table += "|" + "|".join(["---"] * len(headers)) + "|\n"

    for row in answer_rows:
        formatted_row = " | ".join(
            str(value) for value in row
        )
        table += f"| {formatted_row} |\n"

    display(Markdown(table))


def check_td_basic_answers(answer_rows):
    """Check and display the submitted TD-learning calculations."""
    submitted_answers = [
        row[4:] for row in answer_rows
    ]

    has_missing_answers = any(
        value is None
        for row in submitted_answers
        for value in row
    )

    if has_missing_answers:
        print("Complete all table entries before checking your answers.")
        return

    submitted_answers = np.asarray(
        submitted_answers,
        dtype=float,
    )

    if np.allclose(
        submitted_answers,
        TD_BASIC_EXPECTED_ANSWERS,
        rtol=0.0,
        atol=5e-4,
    ):
        print("Congratulations, all answers are correct!")
    else:
        print("Some answers are not correct yet. Review the table.")

    display_table(
        answer_rows,
        TD_BASIC_HEADERS,
    )