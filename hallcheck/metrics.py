"""Evaluation metrics, written out by hand so every number is easy to explain in an interview."""


def binary_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum((not t) and p for t, p in zip(y_true, y_pred))
    fn = sum(t and (not p) for t, p in zip(y_true, y_pred))
    tn = sum((not t) and (not p) for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, len(y_true))
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def auroc(y_true: list[bool], scores: list[float]) -> float:
    """Probability that a random hallucinated answer gets a higher risk score than a clean one."""
    pos = [s for t, s in zip(y_true, scores) if t]
    neg = [s for t, s in zip(y_true, scores) if not t]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def best_threshold(y_true: list[bool], scores: list[float]) -> tuple[float, float]:
    """Risk threshold that maximizes F1. Returns (threshold, f1)."""
    best = (0.5, -1.0)
    for t in sorted(set(scores)):
        f1 = binary_metrics(y_true, [s >= t for s in scores])["f1"]
        if f1 > best[1]:
            best = (t, f1)
    return best


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q / 100 * (len(ordered) - 1))))
    return ordered[idx]
