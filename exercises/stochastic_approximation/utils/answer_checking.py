import numpy as np

from IPython.display import Markdown, display


def check_table_answers(
    answer_rows,
    expected_answers,
    headers,
    answer_start_column=0,
):
    """Check submitted answers and display the completed table."""
    submitted_answers = [
        row[answer_start_column:]
        for row in answer_rows
    ]

    has_missing_answers = any(
        value is None
        for row in submitted_answers
        for value in row
    )

    if has_missing_answers:
        print(
            "Complete all table entries before checking your answers."
        )
        return

    submitted_answers = np.asarray(
        submitted_answers,
        dtype=float,
    )
    expected_answers = np.asarray(
        expected_answers,
        dtype=float,
    )

    if np.allclose(
        submitted_answers,
        expected_answers,
        rtol=0.0,
        atol=5e-4,
    ):
        print("Congratulations! All answers are correct.")
    else:
        print(
            "Some answers are not correct yet. Review the table."
        )

    table = (
        "| " + " | ".join(headers) + " |\n"
        + "| "
        + " | ".join(["---:"] * len(headers))
        + " |\n"
    )

    for row in answer_rows:
        formatted_row = " | ".join(
            str(value) for value in row
        )
        table += f"| {formatted_row} |\n"

    display(Markdown(table))


def check_incremental_mean_answers(answer_rows):
    """Check the answers for the incremental mean exercise."""
    expected_answers = [
        [1.0, -2.0, 2.0],
        [0.5, -2.0, 3.0],
        [1 / 3, 0.0, 3.0],
    ]

    headers = [
        r"$k$",
        r"$x_k$",
        r"$\alpha_k$",
        r"$w_{k-1}-x_k$",
        r"$w_k$",
    ]

    check_table_answers(
        answer_rows=answer_rows,
        expected_answers=expected_answers,
        headers=headers,
        answer_start_column=2,
    )


def check_robbins_monro_answers(answer_rows):
    """Check the answers for the Robbins-Monro exercise."""
    expected_answers = [
        [0.0, 1 / 2, -2.0, 1.0],
        [1.0, 1 / 3, -1.0, 4 / 3],
        [4 / 3, 1 / 4, -2 / 3, 3 / 2],
    ]

    headers = [
        r"$k$",
        r"$w_{k-1}$",
        r"$\alpha_k$",
        r"$g(w_{k-1})$",
        r"$w_k$",
    ]

    check_table_answers(
        answer_rows=answer_rows,
        expected_answers=expected_answers,
        headers=headers,
        answer_start_column=1,
    )


def check_sgd_answers(answer_rows):
    """Check the answers for the SGD exercise."""
    expected_answers = [
        [0.0, 1.0, -2.0, 2.0],
        [2.0, 1 / 2, -2.0, 3.0],
        [3.0, 1 / 3, 0.0, 3.0],
    ]

    headers = [
        r"$k$",
        r"$x_k$",
        r"$w_{k-1}$",
        r"$\alpha_k$",
        r"$\nabla_w f(w_{k-1},x_k)$",
        r"$w_k$",
    ]

    check_table_answers(
        answer_rows=answer_rows,
        expected_answers=expected_answers,
        headers=headers,
        answer_start_column=2,
    )