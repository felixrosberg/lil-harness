"""Small statistics helpers. There is a bug in here for the coding demo to find."""


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean() of an empty list")
    return sum(values) / (len(values) - 1)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
