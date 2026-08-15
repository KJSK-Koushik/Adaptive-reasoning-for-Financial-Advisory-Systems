"""Synthetic finance-math generator.

Every item's question *and* exact answer are produced by the same Python function, so
grading is perfect and there is no annotation noise at all. That matters more here
than it looks: FinQA and TAT-QA carry some label noise, and a reward function built on
noisy labels teaches the DQN to stop at the wrong moments. This source gives the
policy a clean, unambiguous signal to anchor on.

The generators are graded into three intrinsic tiers by the number of arithmetic steps
required:

* ``easy``   - one operation (simple interest, percentage change)
* ``medium`` - two or three operations (compound interest, CAGR, EMI, allocation)
* ``hard``   - four or more, or an iterative/multi-year calculation (SIP with step-up,
  NPV, loan amortisation, blended tax)

Those tiers are only a *prior*. Phase 2 overwrites ``difficulty`` with the label
derived from how the model actually performs, because what is arithmetically short is
not always what a language model finds easy.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from ..config import Config
from ..logging_utils import get_logger
from ..schema import AnswerType, Difficulty, Domain, QARecord
from .text_utils import format_number

log = get_logger("data.synthetic")


@dataclass
class Item:
    question: str
    answer: float
    tier: Difficulty


Generator = Callable[[random.Random], Item]


# --------------------------------------------------------------------------- #
# Easy - single operation
# --------------------------------------------------------------------------- #
def gen_simple_interest(rng: random.Random) -> Item:
    p = rng.randrange(10_000, 500_000, 5_000)
    r = round(rng.uniform(4, 12), 1)
    t = rng.randint(1, 10)
    return Item(
        f"A principal of {p:,} is invested at a simple interest rate of {r}% per annum "
        f"for {t} years. What is the total simple interest earned?",
        p * r * t / 100,
        Difficulty.EASY,
    )


def gen_percentage_change(rng: random.Random) -> Item:
    old = rng.randrange(1_000, 100_000, 100)
    new = round(old * rng.uniform(0.5, 1.8), 0)
    return Item(
        f"A company's quarterly revenue moved from {old:,} to {int(new):,}. "
        f"What was the percentage change? Give your answer as a percentage.",
        (new - old) / old * 100,
        Difficulty.EASY,
    )


def gen_expense_ratio(rng: random.Random) -> Item:
    corpus = rng.randrange(50_000, 2_000_000, 10_000)
    ratio = round(rng.uniform(0.2, 2.5), 2)
    return Item(
        f"A mutual fund holding of {corpus:,} carries an expense ratio of {ratio}% per "
        f"annum. What is the annual fee charged on this holding?",
        corpus * ratio / 100,
        Difficulty.EASY,
    )


# --------------------------------------------------------------------------- #
# Medium - two or three operations
# --------------------------------------------------------------------------- #
def gen_compound_interest(rng: random.Random) -> Item:
    p = rng.randrange(10_000, 500_000, 5_000)
    r = round(rng.uniform(5, 14), 1)
    t = rng.randint(2, 15)
    n = rng.choice([1, 2, 4, 12])
    freq = {1: "annually", 2: "half-yearly", 4: "quarterly", 12: "monthly"}[n]
    amount = p * (1 + r / 100 / n) ** (n * t)
    return Item(
        f"An amount of {p:,} is invested at {r}% per annum compounded {freq} for "
        f"{t} years. What is the maturity value?",
        amount,
        Difficulty.MEDIUM,
    )


def gen_cagr(rng: random.Random) -> Item:
    begin = rng.randrange(50_000, 1_000_000, 10_000)
    years = rng.randint(3, 12)
    end = round(begin * rng.uniform(1.2, 4.0), 0)
    cagr = ((end / begin) ** (1 / years) - 1) * 100
    return Item(
        f"An investment grew from {begin:,} to {int(end):,} over {years} years. "
        f"What is the compound annual growth rate (CAGR)? Give your answer as a percentage.",
        cagr,
        Difficulty.MEDIUM,
    )


def gen_emi(rng: random.Random) -> Item:
    p = rng.randrange(100_000, 5_000_000, 50_000)
    annual = round(rng.uniform(6, 15), 1)
    years = rng.randint(2, 25)
    r, n = annual / 1200, years * 12
    emi = p * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return Item(
        f"A loan of {p:,} is taken at {annual}% annual interest for {years} years, "
        f"repaid in equal monthly instalments. What is the monthly EMI?",
        emi,
        Difficulty.MEDIUM,
    )


def gen_allocation(rng: random.Random) -> Item:
    corpus = rng.randrange(100_000, 5_000_000, 50_000)
    eq, debt = rng.randrange(40, 80, 5), None
    debt = rng.randrange(10, 100 - eq, 5)
    gold = 100 - eq - debt
    r_eq, r_debt, r_gold = (round(rng.uniform(8, 16), 1), round(rng.uniform(4, 8), 1),
                            round(rng.uniform(2, 10), 1))
    blended = (eq * r_eq + debt * r_debt + gold * r_gold) / 100
    return Item(
        f"A portfolio of {corpus:,} is allocated {eq}% to equity, {debt}% to debt and "
        f"{gold}% to gold, returning {r_eq}%, {r_debt}% and {r_gold}% respectively over "
        f"one year. What is the blended portfolio return? Give your answer as a percentage.",
        blended,
        Difficulty.MEDIUM,
    )


def gen_break_even(rng: random.Random) -> Item:
    fixed = rng.randrange(50_000, 2_000_000, 10_000)
    price = rng.randrange(50, 5_000, 10)
    variable = round(price * rng.uniform(0.3, 0.8), 0)
    return Item(
        f"A business has fixed costs of {fixed:,} per year. Each unit sells for {price:,} "
        f"and costs {int(variable):,} to produce. How many units must be sold to break even? "
        f"Round up to the next whole unit.",
        -(-fixed // (price - variable)),
        Difficulty.MEDIUM,
    )


# --------------------------------------------------------------------------- #
# Hard - four or more operations, or iterative
# --------------------------------------------------------------------------- #
def gen_sip(rng: random.Random) -> Item:
    monthly = rng.randrange(1_000, 50_000, 500)
    annual = round(rng.uniform(8, 15), 1)
    years = rng.randint(5, 30)
    i, n = annual / 1200, years * 12
    fv = monthly * (((1 + i) ** n - 1) / i) * (1 + i)
    return Item(
        f"An investor contributes {monthly:,} at the start of every month to a fund "
        f"returning {annual}% per annum, compounded monthly, for {years} years. "
        f"What is the maturity value of this SIP?",
        fv,
        Difficulty.HARD,
    )


def gen_npv(rng: random.Random) -> Item:
    initial = rng.randrange(100_000, 2_000_000, 50_000)
    years = rng.randint(3, 6)
    flows = [rng.randrange(30_000, 700_000, 10_000) for _ in range(years)]
    rate = round(rng.uniform(6, 14), 1)
    npv = -initial + sum(cf / (1 + rate / 100) ** (t + 1) for t, cf in enumerate(flows))
    flow_text = ", ".join(f"year {t + 1}: {cf:,}" for t, cf in enumerate(flows))
    return Item(
        f"A project requires an initial investment of {initial:,} and returns the "
        f"following cash flows ({flow_text}). At a discount rate of {rate}%, what is the "
        f"net present value?",
        npv,
        Difficulty.HARD,
    )


def gen_stepup_sip(rng: random.Random) -> Item:
    monthly = rng.randrange(2_000, 30_000, 1_000)
    annual = round(rng.uniform(9, 14), 1)
    years = rng.randint(5, 15)
    stepup = rng.choice([5, 10, 15])
    i = annual / 1200
    balance, contribution = 0.0, float(monthly)
    for _ in range(years):
        for _ in range(12):
            balance = (balance + contribution) * (1 + i)
        contribution *= 1 + stepup / 100
    return Item(
        f"An investor starts a SIP of {monthly:,} per month in a fund returning "
        f"{annual}% per annum compounded monthly, and increases the monthly contribution "
        f"by {stepup}% at the end of every year. What is the corpus after {years} years?",
        balance,
        Difficulty.HARD,
    )


def gen_tax_slab(rng: random.Random) -> Item:
    # Income steps by 1,000 and a deduction is drawn independently: without both, this
    # generator tops out at a few hundred distinct questions and starves the sampler.
    income = rng.randrange(400_000, 3_000_000, 1_000)
    deduction = rng.randrange(0, 200_000, 5_000)
    taxable = max(0, income - deduction)

    slabs = [(300_000, 0), (600_000, 5), (900_000, 10), (1_200_000, 15),
             (1_500_000, 20), (float("inf"), 30)]
    tax, lower = 0.0, 0
    for upper, rate in slabs:
        if taxable > lower:
            tax += (min(taxable, upper) - lower) * rate / 100
            lower = upper
        else:
            break

    slab_text = ("0% up to 300,000; 5% from 300,001 to 600,000; 10% from 600,001 to "
                 "900,000; 15% from 900,001 to 1,200,000; 20% from 1,200,001 to "
                 "1,500,000; 30% above 1,500,000")
    return Item(
        f"An individual earns {income:,} in a year and claims deductions of "
        f"{deduction:,}. Under a slab system ({slab_text}), what is the total tax "
        f"payable on the taxable income?",
        tax,
        Difficulty.HARD,
    )


def gen_real_return(rng: random.Random) -> Item:
    nominal = round(rng.uniform(6, 16), 1)
    inflation = round(rng.uniform(2, 8), 1)
    tax = rng.choice([0, 10, 15, 20, 30])
    post_tax = nominal * (1 - tax / 100)
    real = ((1 + post_tax / 100) / (1 + inflation / 100) - 1) * 100
    return Item(
        f"An investment returns {nominal}% per annum before tax. Gains are taxed at "
        f"{tax}% and inflation is {inflation}% per annum. What is the real, post-tax "
        f"rate of return? Give your answer as a percentage.",
        real,
        Difficulty.HARD,
    )


GENERATORS: list[Generator] = [
    gen_simple_interest, gen_percentage_change, gen_expense_ratio,
    gen_compound_interest, gen_cagr, gen_emi, gen_allocation, gen_break_even,
    gen_sip, gen_npv, gen_stepup_sip, gen_tax_slab, gen_real_return,
]


#: Consecutive duplicate draws after which a generator is considered exhausted. A
#: generator with a small parameter space would otherwise spin forever and starve the
#: rest of the quota.
_EXHAUSTION_LIMIT = 200


def load(cfg: Config) -> list[QARecord]:
    """Generate ``data.sample_sizes.synthetic`` questions, evenly across generators.

    Each generator gets an equal quota. If one exhausts its parameter space early, its
    shortfall is redistributed over the generators that still have room, so the total
    stays on target instead of silently coming up short.
    """
    n = cfg.data.sample_sizes.get("synthetic") or 0
    if n <= 0:
        return []

    rng = random.Random(cfg.project.seed)
    seen: set[str] = set()
    items: list[Item] = []
    exhausted: set[int] = set()

    def draw(index: int, quota: int) -> int:
        """Draw up to ``quota`` fresh items from one generator. Returns how many."""
        produced, misses = 0, 0
        while produced < quota and misses < _EXHAUSTION_LIMIT:
            item = GENERATORS[index](rng)
            if item.question in seen:
                misses += 1
                continue
            seen.add(item.question)
            items.append(item)
            produced += 1
            misses = 0
        if misses >= _EXHAUSTION_LIMIT:
            exhausted.add(index)
        return produced

    per_generator = max(1, n // len(GENERATORS))
    for i in range(len(GENERATORS)):
        draw(i, per_generator)

    # Redistribute whatever the exhausted generators could not supply.
    rounds = 0
    while len(items) < n and len(exhausted) < len(GENERATORS) and rounds < 20:
        rounds += 1
        available = [i for i in range(len(GENERATORS)) if i not in exhausted]
        shortfall = n - len(items)
        per_each = max(1, shortfall // len(available))
        for i in available:
            if len(items) >= n:
                break
            draw(i, min(per_each, n - len(items)))

    if exhausted:
        log.warning(
            "synthetic: %d generator(s) exhausted their parameter space: %s",
            len(exhausted), sorted(GENERATORS[i].__name__ for i in exhausted),
        )
    if len(items) < n:
        log.warning("synthetic: produced %d of %d requested", len(items), n)

    rng.shuffle(items)
    records = [
        QARecord(
            id=f"synthetic::{i}",
            source="synthetic",
            domain=Domain.INVESTMENT,
            question=item.question,
            gold_answer=format_number(round(float(item.answer), 4)),
            answer_type=AnswerType.NUMERIC,
            # A prior, not a label. `difficulty` stays empty until Phase 2 measures it
            # by sampling the model - see the note on QARecord.difficulty_prior.
            difficulty_prior=item.tier,
        )
        for i, item in enumerate(items)
    ]

    tiers = Counter(str(r.difficulty_prior) for r in records)
    log.info("synthetic: %d records from %d generators, prior tiers=%s",
             len(records), len(GENERATORS), dict(tiers))
    return records
