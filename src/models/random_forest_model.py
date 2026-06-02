from sklearn.ensemble import RandomForestClassifier


def create_random_forest(
    n_estimators=400,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
):
    """
    Create a Random Forest classifier.

    Tree-based models do not require feature standardization.
    """

    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )
