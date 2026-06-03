import copy


class EarlyStopping:
    def __init__(
        self,
        patience=5,
        min_delta=0.0,
        mode="min",
        restore_best_weights=True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.best_state = None
        self.wait = 0

    def _is_improvement(
        self,
        score
    ):
        if self.best_score is None:

            return True

        if self.mode == "min":

            return score < (
                self.best_score
                - self.min_delta
            )

        if self.mode == "max":

            return score > (
                self.best_score
                + self.min_delta
            )

        raise ValueError(
            "Unsupported early stopping mode."
        )

    def step(
        self,
        score,
        model=None
    ):
        if self._is_improvement(
            score
        ):

            self.best_score = score
            self.wait = 0

            if (
                model is not None
                and self.restore_best_weights
            ):

                self.best_state = copy.deepcopy(
                    model.state_dict()
                )

            return False

        self.wait += 1

        return self.wait >= self.patience

    def restore(
        self,
        model
    ):
        if self.best_state is not None:

            model.load_state_dict(
                self.best_state
            )

        return model


def create_early_stopping(
    patience=5,
    min_delta=0.0,
    mode="min",
    restore_best_weights=True
):
    return EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        restore_best_weights=restore_best_weights
    )
