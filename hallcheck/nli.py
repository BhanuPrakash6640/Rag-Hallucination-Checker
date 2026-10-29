"""Step 2 of the checker: a small NLI model that decides if a passage supports a claim.

NLI = natural language inference. Given (premise, hypothesis) it outputs three
probabilities: entailment (premise supports it), contradiction, neutral (unrelated).
"""

# Some models (e.g. sentence-transformers cross-encoders) ship generic LABEL_0.. names.
_KNOWN_LABEL_ORDERS = {
    "cross-encoder/": ["contradiction", "entailment", "neutral"],
}


def _canonical(label: str) -> str:
    label = label.lower()
    if label.startswith("not_entail") or label.startswith("non_entail"):
        return "neutral"
    if "entail" in label:
        return "entailment"
    if "contra" in label:
        return "contradiction"
    if "neutral" in label:
        return "neutral"
    return label


class NLIScorer:
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
        self.labels = self._resolve_labels(model_name)

    def _resolve_labels(self, model_name: str) -> list[str]:
        id2label = self.model.config.id2label
        labels = [_canonical(id2label[i]) for i in range(len(id2label))]
        if all(l.startswith("label_") for l in labels):
            for prefix, order in _KNOWN_LABEL_ORDERS.items():
                if model_name.startswith(prefix) and len(order) == len(labels):
                    return order
            raise ValueError(f"Cannot tell which output of {model_name} means entailment. Labels: {id2label}")
        if "entailment" not in labels:
            raise ValueError(f"{model_name} has no entailment label: {id2label}")
        return labels

    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """pairs = [(premise/evidence, hypothesis/claim)] -> [{"entailment": p, "contradiction": p, "neutral": p}]"""
        torch = self.torch
        results: list[dict[str, float] | None] = [None] * len(pairs)
        # Sort by length so each batch has similar lengths (less padding = faster).
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        for start in range(0, len(order), self.batch_size):
            batch_ids = order[start : start + self.batch_size]
            premises = [pairs[i][0] for i in batch_ids]
            hypotheses = [pairs[i][1] for i in batch_ids]
            inputs = self.tokenizer(
                premises, hypotheses, truncation="only_first", max_length=self.max_length,
                padding=True, return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self.model(**inputs).logits.float(), dim=-1).cpu().tolist()
            for i, row in zip(batch_ids, probs):
                scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
                for label, p in zip(self.labels, row):
                    scores[label] = scores.get(label, 0.0) + p
                results[i] = scores
        return results  # type: ignore[return-value]
