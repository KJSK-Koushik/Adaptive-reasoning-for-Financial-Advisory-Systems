"""Statlog German Credit adapter - 1,000 applicants templated into risk questions.

The raw file is space-separated with 20 attributes plus a label (1 = good risk,
2 = bad risk). Attributes are opaque codes (``A11``, ``A34``, ...) decoded below from
the accompanying ``german.doc``.

**Protected attributes.** Attribute 9 encodes personal status *and sex*, and attribute
20 is foreign-worker status. Templating those into a credit-approval question would
bake sex- and nationality-based lending into the training data and into whatever the
model learns to justify. With ``data.exclude_protected_attributes`` set (the default)
both are omitted from the rendered question; the marital-status portion of attribute 9
is kept, since it is a legitimate financial signal. The flag exists so the choice is
visible and reversible rather than silent.
"""

from __future__ import annotations

import random

from ... import paths
from ...config import Config
from ...logging_utils import get_logger
from ...schema import AnswerType, Domain, QARecord

log = get_logger("data.german_credit")

CHECKING = {
    "A11": "less than 0 DM", "A12": "between 0 and 200 DM",
    "A13": "at least 200 DM, with salary assigned for at least a year",
    "A14": "no checking account",
}
HISTORY = {
    "A30": "no credits taken, or all credits paid back duly",
    "A31": "all credits at this bank paid back duly",
    "A32": "existing credits paid back duly until now",
    "A33": "delays in paying off credit in the past",
    "A34": "a critical account, or other credits existing elsewhere",
}
PURPOSE = {
    "A40": "a new car", "A41": "a used car", "A42": "furniture or equipment",
    "A43": "radio or television", "A44": "domestic appliances", "A45": "repairs",
    "A46": "education", "A47": "a vacation", "A48": "retraining", "A49": "business",
    "A410": "other purposes",
}
SAVINGS = {
    "A61": "less than 100 DM", "A62": "between 100 and 500 DM",
    "A63": "between 500 and 1000 DM", "A64": "1000 DM or more",
    "A65": "unknown or no savings account",
}
EMPLOYMENT = {
    "A71": "unemployed", "A72": "less than 1 year", "A73": "1 to 4 years",
    "A74": "4 to 7 years", "A75": "7 years or more",
}
MARITAL = {
    "A91": "divorced or separated", "A92": "divorced, separated or married",
    "A93": "single", "A94": "married or widowed", "A95": "single",
}
DEBTORS = {"A101": "none", "A102": "a co-applicant", "A103": "a guarantor"}
PROPERTY = {
    "A121": "real estate", "A122": "building society savings or life insurance",
    "A123": "a car or other property", "A124": "no known property",
}
INSTALLMENT_PLANS = {"A141": "at another bank", "A142": "at stores", "A143": "none"}
HOUSING = {"A151": "rented", "A152": "owned", "A153": "provided free"}
JOB = {
    "A171": "unemployed or an unskilled non-resident",
    "A172": "an unskilled resident",
    "A173": "a skilled employee or official",
    "A174": "in management, self-employed, or highly qualified",
}
TELEPHONE = {"A191": "no registered telephone", "A192": "a registered telephone"}

_QUESTIONS = [
    "Is this applicant a good or bad credit risk?",
    "Based on this credit profile, should the applicant be classified as a good or "
    "bad credit risk?",
    "Assess this loan application and classify the applicant's credit risk as good or bad.",
]


def _describe(f: list[str], cfg: Config) -> str:
    """Render one applicant record as prose."""
    duration, amount, rate = int(f[1]), int(f[4]), int(f[7])
    residence, age, n_credits, dependants = int(f[10]), int(f[12]), int(f[15]), int(f[17])

    parts = [
        f"An applicant aged {age} is requesting credit of {amount:,} DM over "
        f"{duration} months for {PURPOSE.get(f[3], 'an unspecified purpose')}.",
        f"Their checking account status is {CHECKING.get(f[0], 'unknown')} and their "
        f"savings are {SAVINGS.get(f[5], 'unknown')}.",
        f"Credit history: {HISTORY.get(f[2], 'unknown')}.",
        f"They have been employed for {EMPLOYMENT.get(f[6], 'an unknown period')} and "
        f"work as {JOB.get(f[16], 'an unspecified occupation')}.",
        f"The instalment rate is {rate}% of disposable income.",
        f"Their housing is {HOUSING.get(f[14], 'unknown')} and they own "
        f"{PROPERTY.get(f[11], 'unknown property')}.",
        f"They have lived at their current address for {residence} years, hold "
        f"{n_credits} existing credit(s) at this bank, and support {dependants} dependant(s).",
        f"Other instalment plans: {INSTALLMENT_PLANS.get(f[13], 'unknown')}. "
        f"Other debtors or guarantors: {DEBTORS.get(f[9], 'none')}.",
        f"The applicant has {TELEPHONE.get(f[18], 'an unknown telephone status')}.",
    ]

    if not cfg.data.exclude_protected_attributes:
        parts.append(f"Personal status: {MARITAL.get(f[8], 'unknown')}.")
        parts.append("Foreign worker: " + ("yes." if f[19] == "A201" else "no."))

    return " ".join(parts)


def load(cfg: Config) -> list[QARecord]:
    path = paths.RAW_SOURCES["german_credit"] / "german.data"
    if not path.exists():
        log.warning("german.data not found in %s (run the download step)", path.parent)
        return []

    rng = random.Random(cfg.project.seed)
    records: list[QARecord] = []

    for i, line in enumerate(path.read_text(encoding="latin-1").splitlines()):
        fields = line.split()
        if len(fields) != 21:
            continue
        label = "good" if fields[20] == "1" else "bad"
        records.append(
            QARecord(
                id=f"german_credit::{i}",
                source="german_credit",
                domain=Domain.RISK,
                question=rng.choice(_QUESTIONS),
                context=_describe(fields, cfg),
                gold_answer=label,
                answer_type=AnswerType.CATEGORICAL,
                answer_options=["good", "bad"],
            )
        )

    n_bad = sum(1 for r in records if r.gold_answer == "bad")
    log.info("german_credit: %d records (%d bad risk)", len(records), n_bad)
    return records
