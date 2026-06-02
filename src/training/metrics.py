from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)


def compute_accuracy(
    y_true,
    y_pred
):
    return accuracy_score(
        y_true,
        y_pred
    )


def compute_macro_f1(
    y_true,
    y_pred
):
    return f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


def compute_weighted_f1(
    y_true,
    y_pred
):
    return f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


def classification_report_dict(
    y_true,
    y_pred,
    labels=None,
    target_names=None
):
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
