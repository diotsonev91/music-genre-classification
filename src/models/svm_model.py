from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def create_svm(
    kernel="rbf",
    c=10.0,
    gamma="scale",
    class_weight="balanced",
    random_state=42
):
    """
    Create an SVM classifier pipeline.

    SVM models are sensitive to feature scale, so the returned
    estimator includes a StandardScaler fitted only during training.
    """

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel=kernel,
                C=c,
                gamma=gamma,
                class_weight=class_weight,
                random_state=random_state
            )
        )
    ])
