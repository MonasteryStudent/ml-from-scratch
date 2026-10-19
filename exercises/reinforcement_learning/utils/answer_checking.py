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

SARSA_EXPECTED_ANSWERS = np.array([
    [0.0, 0.0, 1.0, -1.0, 0.5],
    [0.0, 0.0, 2.0, -2.0, 1.0],
])

SARSA_HEADERS = [
    r"$t$",
    r"$(s_t,a_t)$",
    r"$r_{t+1}$",
    r"$q_t(s_t,a_t)$",
    r"$q_t(s_{t+1},a_{t+1})$",
    r"$\bar{q}_t$",
    r"$\delta_t$",
    r"$q_{t+1}(s_t,a_t)$",
]

SARSA_POLICY_EXPECTED_ANSWERS = np.array([
    [0.0, 0.5, 1.0, 0.1, 0.9],
    [1.0, 0.0, 0.0, 0.9, 0.1],
])

SARSA_POLICY_HEADERS = [
    "State",
    r"$q(s,a_0)$",
    r"$q(s,a_1)$",
    "Greedy action",
    r"$\pi(a_0\mid s)$",
    r"$\pi(a_1\mid s)$",
]

Q_LEARNING_EXPECTED_ANSWERS = np.array([
    [0.0, 2.0, 2.0, -2.0, 1.0],
    [0.0, 0.0, 0.0,  0.0, 0.0],
    [1.0, 2.0, 2.0, -1.0, 1.5],
    [2.0, 0.0, 4.0, -2.0, 3.0],
])

Q_LEARNING_HEADERS = [
    "Update",
    "Episode",
    r"$(s_t,a_t)$",
    r"$r_{t+1}$",
    r"$q_t(s_t,a_t)$",
    r"$\max_a q_t(s_{t+1},a)$",
    r"$\bar{q}_t$",
    r"$\delta_t$",
    r"$q_{t+1}(s_t,a_t)$",
]

Q_LEARNING_POLICY_EXPECTED_ANSWERS = np.array([
    [0.0, 1.5, 1.0, 0.0, 1.0],
    [3.0, 0.0, 0.0, 1.0, 0.0],
])

Q_LEARNING_POLICY_HEADERS = [
    "State",
    r"$q(s,a_0)$",
    r"$q(s,a_1)$",
    "Greedy action",
    r"$\pi_T(a_0\mid s)$",
    r"$\pi_T(a_1\mid s)$",
]


def display_table(answer_rows, headers):
    """Display submitted answers as a Markdown table."""
    table = "| " + " | ".join(headers) + " |\n"
    table += "|" + "|".join(["---"] * len(headers)) + "|\n"

    for row in answer_rows:
        formatted_row = " | ".join(
            str(value) for value in row
        )
        table += f"| {formatted_row} |\n"

    display(Markdown(table))


def extract_answers(rows, first_answer_column):
    """Return submitted numeric answers or None if entries are missing."""
    answers = [
        row[first_answer_column:]
        for row in rows
    ]

    has_missing_answers = any(
        value is None
        for row in answers
        for value in row
    )

    if has_missing_answers:
        return None

    return np.asarray(
        answers,
        dtype=float,
    )


def answers_match(submitted_answers, expected_answers):
    """Return whether submitted and expected answers match."""
    return np.allclose(
        submitted_answers,
        expected_answers,
        rtol=0.0,
        atol=5e-4,
    )


def check_answer_tables(table_definitions):
    """Check and display one or more exercise tables."""
    checked_tables = []

    for table_definition in table_definitions:
        submitted_answers = extract_answers(
            table_definition["rows"],
            table_definition["first_answer_column"],
        )

        if submitted_answers is None:
            print(
                "Complete all table entries before checking your answers."
            )
            return

        checked_tables.append(
            (
                table_definition,
                submitted_answers,
            )
        )

    all_answers_are_correct = all(
        answers_match(
            submitted_answers,
            table_definition["expected_answers"],
        )
        for table_definition, submitted_answers in checked_tables
    )

    if all_answers_are_correct:
        print("Congratulations, all answers are correct!")
    else:
        print(
            "Some answers are not correct yet. Review your entries."
        )

    for table_definition, _ in checked_tables:
        title = table_definition.get("title")

        if title is not None:
            display(Markdown(f"#### {title}"))

        display_table(
            table_definition["rows"],
            table_definition["headers"],
        )


def check_td_basic_answers(answer_rows):
    """Check and display the submitted TD-learning calculations."""
    check_answer_tables([
        {
            "rows": answer_rows,
            "first_answer_column": 4,
            "expected_answers": TD_BASIC_EXPECTED_ANSWERS,
            "headers": TD_BASIC_HEADERS,
        },
    ])


def check_sarsa_answers(
    answer_rows,
    policy_rows,
):
    """Check and display the submitted Sarsa calculations."""
    check_answer_tables([
        {
            "title": "Action-value updates",
            "rows": answer_rows,
            "first_answer_column": 3,
            "expected_answers": SARSA_EXPECTED_ANSWERS,
            "headers": SARSA_HEADERS,
        },
        {
            "title": "Improved policy",
            "rows": policy_rows,
            "first_answer_column": 1,
            "expected_answers": SARSA_POLICY_EXPECTED_ANSWERS,
            "headers": SARSA_POLICY_HEADERS,
        },
    ])


def check_q_learning_answers(
    answer_rows,
    policy_rows,
):
    """Check and display the submitted Q-learning calculations."""
    check_answer_tables([
        {
            "title": "Q-learning updates",
            "rows": answer_rows,
            "first_answer_column": 4,
            "expected_answers": Q_LEARNING_EXPECTED_ANSWERS,
            "headers": Q_LEARNING_HEADERS,
        },
        {
            "title": "Target policy",
            "rows": policy_rows,
            "first_answer_column": 1,
            "expected_answers": Q_LEARNING_POLICY_EXPECTED_ANSWERS,
            "headers": Q_LEARNING_POLICY_HEADERS,
        },
    ])