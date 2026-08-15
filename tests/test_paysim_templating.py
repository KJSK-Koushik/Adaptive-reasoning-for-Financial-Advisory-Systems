"""Templating tests for PaySim.

The rendered text is the model's entire view of the transaction, so a wrong or
spurious sentence here becomes a wrong reward in Phase 5. These cover the two bugs
found when the adapter first ran against the real CSV.
"""

from __future__ import annotations

import random

import pytest

from adaptive_reasoning.data.adapters.paysim import _describe


def _row(**overrides) -> dict:
    row = {
        "step": 14 * 24 + 4,
        "type": "TRANSFER",
        "amount": 72_665.95,
        "nameOrig": "C1234567",
        "oldbalanceOrg": 72_665.95,
        "newbalanceOrig": 0.0,
        "nameDest": "C7654321",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 1,
    }
    row.update(overrides)
    return row


def test_merchant_destination_reports_no_balances():
    """PaySim never tracks merchant balances; printing 0.00 would state a falsehood."""
    text = _describe(_row(type="PAYMENT", nameDest="M123", amount=13_009.45), random.Random(0))
    assert "merchant account" in text
    assert "before and" not in text.split("merchant account")[1]


def test_merchant_payment_omits_the_unchanged_balance_observation():
    """It fired on every merchant payment, leaking transaction type, not fraud."""
    text = _describe(_row(type="PAYMENT", nameDest="M123", amount=13_009.45), random.Random(0))
    assert "did not change" not in text


def test_customer_transfer_keeps_the_unchanged_balance_observation():
    text = _describe(_row(), random.Random(0))
    assert "did not change despite receiving the funds" in text
    assert "transfer." not in text.split("did not change")[1]  # wording no longer says "transfer"


def test_emptied_account_is_reported():
    text = _describe(_row(), random.Random(0))
    assert "entire balance" in text


def test_partial_debit_is_not_reported_as_emptied():
    text = _describe(_row(amount=100.0, oldbalanceOrg=5000.0, newbalanceOrig=4900.0),
                     random.Random(0))
    assert "entire balance" not in text


def test_sentence_order_actually_varies():
    """`rng.shuffle(lines[1:3])` shuffled a copy and was a silent no-op."""
    texts = {_describe(_row(), random.Random(seed)) for seed in range(30)}
    orders = {t.index("The originating") < t.index("The destination") for t in texts}
    assert orders == {True, False}, "origin/destination order never varies"


def test_no_fraud_label_leaks_into_the_text():
    for seed in range(20):
        for is_fraud in (0, 1):
            text = _describe(_row(isFraud=is_fraud), random.Random(seed))
            assert "fraud" not in text.lower()
            assert "flag" not in text.lower()


@pytest.mark.parametrize("tx_type", ["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"])
def test_every_transaction_type_renders(tx_type):
    text = _describe(_row(type=tx_type), random.Random(1))
    assert text.startswith("On day")
    assert len(text) > 80


def test_hour_and_day_derive_from_step():
    text = _describe(_row(step=49), random.Random(0))   # 49 = day 3, hour 01
    assert "day 3 at 01:00" in text
