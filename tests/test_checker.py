"""Fast tests that need no downloads: a tiny fake NLI model stands in for DeBERTa."""

import math

from evaluate import flatten
from hallcheck.claims import extract_claims
from hallcheck.loader import chunk_text
from hallcheck.metrics import auroc, best_threshold, binary_metrics
from hallcheck.numbers import extract_numbers, missing_numbers
from hallcheck.text_utils import keywords, split_sentences
from hallcheck.verifier import CONTRADICTED, SKIPPED, SUPPORTED, UNSUPPORTED, HallucinationChecker


class FakeNLI:
    """Entailment if every keyword of the claim is in the premise; contradiction on a 'not' mismatch."""

    @staticmethod
    def _one(sentence, claim):
        pk, ck = keywords(sentence), keywords(claim)
        neg_p, neg_c = " not " in f" {sentence.lower()} ", " not " in f" {claim.lower()} "
        overlap = len(ck & pk) / max(1, len(ck))
        if overlap == 1.0 and neg_p == neg_c:
            return "entail"
        if overlap >= 0.5 and neg_p != neg_c:
            return "contra"
        return "neutral"

    def predict(self, pairs):
        out = []
        for premise, claim in pairs:
            labels = {self._one(s, claim) for s in split_sentences(premise)}
            if "entail" in labels:
                out.append({"entailment": 0.95, "contradiction": 0.02, "neutral": 0.03})
            elif "contra" in labels:
                out.append({"entailment": 0.03, "contradiction": 0.9, "neutral": 0.07})
            else:
                out.append({"entailment": 0.05, "contradiction": 0.05, "neutral": 0.9})
        return out


SOURCE = (
    "Full-time employees receive 18 days of paid leave per year. "
    "Unused paid leave does not carry over to the next year. "
    "Employees receive 10 days of sick leave per year."
)


# ---------------- text utilities
def test_split_sentences_keeps_abbreviations_and_decimals():
    text = "Leave is 1.5 days a month, e.g. in January. Dr. Rao approves it.\n- Bullet point one"
    assert split_sentences(text) == [
        "Leave is 1.5 days a month, e.g. in January.",
        "Dr. Rao approves it.",
        "Bullet point one",
    ]


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} has exactly seven words." for i in range(20))
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    assert len(chunks) > 3
    assert all(len(c.split()) <= 30 for c in chunks)
    assert chunks[0].split(".")[-2].strip() in chunks[1]  # last sentence repeated as overlap


# ---------------- claims
def test_extract_claims_splits_compound_and_strips_citations():
    claims = extract_claims("You get 18 paid leave days [1], and unused days carry over to next year [2].")
    assert [c.text for c in claims] == ["You get 18 paid leave days.", "Unused days carry over to next year."]


def test_extract_claims_skips_abstentions_and_filler():
    claims = extract_claims(
        "According to the documents, interns work from the office. "
        "The documents do not mention internet bills. I hope this helps!"
    )
    assert claims[0].text == "Interns work from the office." and not claims[0].skipped
    assert claims[1].skipped and claims[2].skipped


def test_short_list_items_are_not_split():
    claims = extract_claims("The allowance covers a chair, desk, and monitor.")
    assert len(claims) == 1


# ---------------- numbers
def test_number_extraction_normalizes_formats():
    assert extract_numbers("Rs. 15,000 for 1.50 days, ten people, 0.5 hours, 2026-10-29") == {
        "15000", "1.5", "10", "0.5", "2026", "29"
    }
    assert extract_numbers("one of the benefits") == set()


def test_missing_numbers_catches_wrong_year():
    assert missing_numbers("She died on February 7, 2022.", "She died on February 7, 1945.") == ["2022"]
    assert missing_numbers("Ten days of sick leave.", "10 days of sick leave") == []


# ---------------- verifier
def test_checker_labels_each_claim():
    checker = HallucinationChecker(FakeNLI())
    answer = (
        "Employees receive 18 days of paid leave per year, and unused paid leave does carry over. "
        "Employees receive 12 days of sick leave per year. I hope this helps!"
    )
    result = checker.check(answer, [SOURCE])
    verdicts = [c.verdict for c in result.claims]
    assert verdicts[0] == SUPPORTED
    assert verdicts[1] in (CONTRADICTED, UNSUPPORTED)
    assert verdicts[2] == UNSUPPORTED and result.claims[2].missing_numbers == ["12"]
    assert verdicts[3] == SKIPPED
    assert result.hallucinated and result.risk_score >= 0.9


def test_checker_passes_grounded_answer():
    checker = HallucinationChecker(FakeNLI())
    result = checker.check("Employees receive 10 days of sick leave per year [3].", [SOURCE])
    assert not result.hallucinated
    assert result.claims[0].evidence and "sick leave" in result.claims[0].evidence
    assert result.to_dict()["counts"][SUPPORTED] == 1


def test_checker_handles_empty_inputs():
    checker = HallucinationChecker(FakeNLI())
    assert checker.check("", [SOURCE]).claims == []
    result = checker.check("Employees receive 18 days of paid leave.", [""])
    assert result.claims[0].verdict == UNSUPPORTED


def test_checker_accepts_retrieved_chunk_dicts():
    checker = HallucinationChecker(FakeNLI())
    result = checker.check("Employees receive 10 days of sick leave per year.", [{"text": SOURCE, "source": "leave.md"}])
    assert result.claims[0].evidence_source == "leave.md"


# ---------------- evaluation helpers
def test_metrics():
    m = binary_metrics([True, True, False, False], [True, False, True, False])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert auroc([True, False], [0.9, 0.1]) == 1.0
    assert math.isnan(auroc([True], [0.5]))
    thr, f1 = best_threshold([True, True, False], [0.9, 0.8, 0.2])
    assert f1 == 1.0 and 0.2 < thr <= 0.8


def test_flatten_data2txt_record():
    lines = flatten({"name": "Cafe X", "hours": {"Monday": "9-5"}, "reviews": [{"stars": 4}]})
    assert lines == ["name: Cafe X.", "hours Monday: 9-5.", "reviews stars: 4."]


def test_list_with_and_is_not_split_and_meta_sentences_skipped():
    claims = extract_claims(
        "Sure! Here's a summary of the article in 105 words or less:\n"
        "Gov. Newsom will visit Hong Kong, Beijing, Jiangsu, and Shanghai, where he will meet leaders."
    )
    checked = [c.text for c in claims if not c.skipped]
    assert checked == ["Gov. Newsom will visit Hong Kong, Beijing, Jiangsu, and Shanghai, where he will meet leaders."]
