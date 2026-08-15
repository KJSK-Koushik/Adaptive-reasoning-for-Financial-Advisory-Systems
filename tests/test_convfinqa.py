"""ConvFinQA adapter tests.

The central one is ``test_per_turn_answers_come_from_annotation_not_qa``: the ``qa``
block is inherited from the source FinQA example and is identical across every turn,
so grading against it would silently mislabel the entire dataset.
"""

from __future__ import annotations

import pytest

from adaptive_reasoning.config import load_config
from adaptive_reasoning.data.adapters.convfinqa import _records, _turns

# Modelled on dev.json[0] / dev_turn.json[0..4] (Single_MRO/2007/page_134.pdf-1).
DIALOGUE = [
    "what was the weighted average exercise price per share in 2007?",
    "and what was it in 2005?",
    "what was, then, the change over the years?",
    "what was the weighted average exercise price per share in 2005?",
    "and how much does that change represent in relation to this 2005 value?",
]
TURN_ANSWERS = [60.94, 25.14, 35.8, 25.14, 1.42403]


def _conversation_item() -> dict:
    return {
        "id": "Single_MRO/2007/page_134.pdf-1",
        "pre_text": ["some narrative"],
        "post_text": [],
        "table": [["year", "price"], ["2007", "60.94"], ["2005", "25.14"]],
        "qa": {
            "question": "by how much did the price increase from 2005 to 2007?",
            "exe_ans": 1.42403,          # the ORIGINAL question's answer
            "gold_inds": {"table_1": "2007 price 60.94", "table_2": "2005 price 25.14"},
        },
        "annotation": {
            "dialogue_break": DIALOGUE,
            "exe_ans_list": TURN_ANSWERS,
        },
    }


def _turn_item(turn_ind: int) -> dict:
    item = _conversation_item()
    item["id"] = f"Single_MRO/2007/page_134.pdf-1_{turn_ind}"
    item["annotation"] = {
        "cur_dial": DIALOGUE[: turn_ind + 1],
        "exe_ans": TURN_ANSWERS[turn_ind],
        "exe_ans_list": TURN_ANSWERS,
    }
    return item


@pytest.fixture
def cfg():
    return load_config()


def test_conversation_layout_yields_one_record_per_turn(cfg):
    records = _records(_conversation_item(), cfg, 0)
    assert len(records) == len(DIALOGUE)


def test_per_turn_answers_come_from_annotation_not_qa(cfg):
    """qa.exe_ans is 1.42403 for every turn; the real answers differ per turn."""
    records = _records(_conversation_item(), cfg, 0)
    golds = [r.gold_answer for r in records]
    assert golds == ["60.94", "25.14", "35.8", "25.14", "1.42403"]
    # If the qa block had been used, every gold would be the same value.
    assert len(set(golds)) > 1


def test_turn_layout_uses_its_own_answer(cfg):
    """Turn-level items must resolve to that turn's answer, not the final one."""
    for turn_ind in range(len(DIALOGUE)):
        records = _records(_turn_item(turn_ind), cfg, 0)
        assert records, f"turn {turn_ind} produced nothing"
        assert records[-1].question.startswith(DIALOGUE[turn_ind][:20])
        assert records[-1].gold_answer == str(TURN_ANSWERS[turn_ind]).rstrip("0").rstrip(".")


def test_history_is_prepended_so_later_turns_are_answerable(cfg):
    records = _records(_conversation_item(), cfg, 0)
    first, last = records[0], records[-1]
    assert "Earlier in this conversation" not in first.context
    assert "Earlier in this conversation" in last.context
    # The referent of "that change" must be present.
    assert "35.8" in last.context


def test_first_turn_has_no_history(cfg):
    records = _records(_conversation_item(), cfg, 0)
    assert "Q:" not in records[0].context


def test_turns_returns_empty_without_answers(cfg):
    """test_private.json ships dialogue with no answers and must be skipped."""
    item = _conversation_item()
    item["annotation"] = {"dialogue_break": DIALOGUE}      # no exe_ans_list
    assert _turns(item) == []
    assert _records(item, cfg, 0) == []


def test_mismatched_answer_list_is_rejected(cfg):
    item = _conversation_item()
    item["annotation"]["exe_ans_list"] = TURN_ANSWERS[:2]   # fewer answers than turns
    assert _turns(item) == []


def test_yes_no_answers_become_categorical(cfg):
    item = _conversation_item()
    item["annotation"]["dialogue_break"] = ["did revenue increase?"]
    item["annotation"]["exe_ans_list"] = ["yes"]
    records = _records(item, cfg, 0)
    assert records[0].answer_type == "categorical"
    assert records[0].gold_answer == "yes"
    assert records[0].answer_options == ["yes", "no"]


def test_ids_are_unique_per_turn(cfg):
    records = _records(_conversation_item(), cfg, 0)
    assert len({r.id for r in records}) == len(records)
