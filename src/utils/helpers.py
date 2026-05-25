from pathlib import Path


def ensure_dir(path):

    Path(
        path
    ).mkdir(
        parents=True,
        exist_ok=True
    )


def print_separator():

    print(
        "=" * 40
    )