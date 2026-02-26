from typing import List

def aggregate_scores(scores: List[float], mode: str = "mean") -> float:
    if not scores:
        return 0.0
    if mode == "max":
        return float(max(scores))
    return float(sum(scores) / len(scores))