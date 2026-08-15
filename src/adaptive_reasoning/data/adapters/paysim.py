"""PaySim adapter - mobile money transactions templated into fraud questions.

The raw file is 6.3M rows (~470 MB) with only 8,213 fraud cases (0.13%). Two
consequences shaped this module:

* **Memory.** The CSV is read in chunks. Every fraud row is retained (they are rare
  and precious) while legitimate rows are reservoir-sampled, so peak memory stays flat
  regardless of file size.
* **Balance.** At the native 0.13% fraud rate, "always answer no" scores 99.87% and the
  reward signal carries no information. ``data.paysim_fraud_ratio`` (default 0.35)
  rebalances the sample.

Columns deliberately excluded from the rendered question:

* ``isFlaggedFraud`` - the simulator's own fraud flag. Leaking it would make the task
  trivial and the results meaningless.
* ``nameOrig``/``nameDest`` account IDs - except the leading character, which encodes
  customer (``C``) vs merchant (``M``) and is genuinely informative.
"""

from __future__ import annotations

import random

from ... import paths
from ...config import Config
from ...logging_utils import get_logger
from ...schema import AnswerType, Domain, QARecord
from ..download import check_manual

log = get_logger("data.paysim")

USE_COLS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud",
]

_TYPE_PHRASE = {
    "TRANSFER": "a transfer",
    "CASH_OUT": "a cash withdrawal",
    "CASH_IN": "a cash deposit",
    "PAYMENT": "a payment",
    "DEBIT": "a debit",
}

_QUESTIONS = [
    "Is this transaction fraudulent?",
    "Based on the account balances and the transaction pattern, is this transaction "
    "fraudulent?",
    "Assess whether this mobile money transaction should be flagged as fraudulent.",
]


#: PaySim only tracks destination balances for customer accounts. Merchant ("M")
#: destinations are always 0.00 before and after, whatever the transaction, so
#: reporting them as observed balances would be untrue and would leak the
#: transaction type rather than any fraud signal.
_DEST_BALANCE_TRACKED = {"TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"}


def _describe(row: dict, rng: random.Random) -> str:
    """Render one transaction as a short natural-language account."""
    step = int(row["step"])
    hour, day = step % 24, step // 24 + 1
    tx_type = str(row["type"])
    kind = _TYPE_PHRASE.get(tx_type, f"a {tx_type.lower()} transaction")
    is_merchant = str(row["nameDest"]).startswith("M")

    amount = float(row["amount"])
    old_o, new_o = float(row["oldbalanceOrg"]), float(row["newbalanceOrig"])
    old_d, new_d = float(row["oldbalanceDest"]), float(row["newbalanceDest"])

    opening = f"On day {day} at {hour:02d}:00, {kind} of {amount:,.2f} was requested."
    origin = (
        f"The originating account held {old_o:,.2f} before the transaction and "
        f"{new_o:,.2f} after it."
    )
    if is_merchant:
        destination = "The destination is a merchant account."
    else:
        destination = (
            f"The destination is another customer account, which held {old_d:,.2f} "
            f"before and {new_d:,.2f} after."
        )

    # Vary the order of the two balance sentences so the model cannot lean on a fixed
    # sentence position. Slicing a list returns a copy, so this must be shuffled as its
    # own list and reassembled - shuffling `lines[1:3]` in place is a silent no-op.
    middle = [origin, destination]
    rng.shuffle(middle)

    parts = [opening, *middle]

    # Derived observations, phrased as facts rather than hints, so the model still has
    # to work out what they mean. Only stated where they are actually meaningful.
    if old_o > 0 and abs(amount - old_o) < 0.01:
        parts.append("The transaction amount equals the originating account's entire balance.")
    if (
        not is_merchant
        and tx_type in _DEST_BALANCE_TRACKED
        and amount > 0
        and abs(new_d - old_d) < 0.01
    ):
        parts.append(
            "The destination account balance did not change despite receiving the funds."
        )

    return " ".join(parts)


def load(cfg: Config) -> list[QARecord]:
    path = check_manual("paysim")
    if path is None:
        log.warning(
            "PaySim CSV not found in %s - skipping. Download from "
            "https://www.kaggle.com/datasets/ealaxi/paysim1",
            paths.RAW_SOURCES["paysim"],
        )
        return []

    import pandas as pd

    target = cfg.data.sample_sizes.get("paysim") or 5000
    n_fraud_target = int(target * cfg.data.paysim_fraud_ratio)
    n_legit_target = target - n_fraud_target

    rng = random.Random(cfg.project.seed)
    fraud_rows: list[dict] = []
    legit_rows: list[dict] = []
    seen_legit = 0

    log.info("scanning %s in chunks", path.name)
    for chunk in pd.read_csv(path, usecols=USE_COLS, chunksize=500_000):
        fraud_rows.extend(chunk[chunk.isFraud == 1].to_dict("records"))

        # Reservoir sampling over the legitimate rows: constant memory, uniform sample.
        for record in chunk[chunk.isFraud == 0].to_dict("records"):
            seen_legit += 1
            if len(legit_rows) < n_legit_target:
                legit_rows.append(record)
            else:
                j = rng.randrange(seen_legit)
                if j < n_legit_target:
                    legit_rows[j] = record

    log.info("paysim: %d fraud rows found, %d legitimate rows scanned", len(fraud_rows), seen_legit)

    if len(fraud_rows) > n_fraud_target:
        fraud_rows = rng.sample(fraud_rows, n_fraud_target)

    selected = fraud_rows + legit_rows
    rng.shuffle(selected)

    records = [
        QARecord(
            id=f"paysim::{i}",
            source="paysim",
            domain=Domain.FRAUD,
            question=rng.choice(_QUESTIONS),
            context=_describe(row, rng),
            gold_answer="yes" if int(row["isFraud"]) == 1 else "no",
            answer_type=AnswerType.CATEGORICAL,
            answer_options=["yes", "no"],
        )
        for i, row in enumerate(selected)
    ]

    n_fraud = sum(1 for r in records if r.gold_answer == "yes")
    log.info("paysim: %d records (%d fraud, %.0f%%)", len(records), n_fraud,
             100 * n_fraud / max(len(records), 1))
    return records
