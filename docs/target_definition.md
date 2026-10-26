# Target definition (Final decisions)

## Data (established, from profiling)
- Freddie Mac Single-Family Loan-Level Dataset, **sample file, vintage 2015**: 50,000 loans, all with loan ID prefix `F15`.
- Standard dataset: all loans are fixed-rate (`FRM`).
- Performance data runs 2015-01 to **2026-03** (3,467,166 loan-month rows), so every loan has long follow-up.
- Fields unusable for this vintage: `valuation_method` and `vantage_score` (100% blank or "not available"); `special_program` is 99% blank.

## Prediction rule (PROPOSED): loan age 24
- One prediction month per loan: the 24th month after the first payment month (age = period - first payment month + 1).
- Outcome window: the 12 months after the prediction month.
- Candidate rules were compared on the whole sample (counts from `window_counts.py`):

| Rule | Eligible | Defaults in window | Rate |
|---|---|---|---|
| age 12 | 45,358 | 196 | 0.43% |
| **age 24** | **40,556** | **221** | **0.54%** |
| age 36 | 36,571 | 162 | 0.44% |
| calendar 2017-12 | 38,802 | 175 | 0.45% |
| calendar 2018-12 | 35,592 | 160 | 0.45% |
| calendar 2019-12 | 32,195 | 1,199 | 3.72% (window is entirely inside the COVID period; not comparable) |

- Reason for age 24: most events among the rules whose windows do not overlap March 2020 to December 2021.
- Consequence: the model describes loans that are **still performing at month 24**, a survivor population (about 9,400 of the 50,000 loans have already left or are ineligible at that point). It says nothing about earlier-life default.

## Eligible population at the prediction month (PROPOSED)
- A performance row exists for the prediction month.
- Delinquency status is 00, 01, or 02 (less than 90 days) and the loan has not terminated.
- Current balance is greater than zero.
- Loans with first payment after 2016-03 (at least 76 loans: converted or modified-type records) are dropped; the exact count is reported in the data funnel.

## Default event (PROPOSED)
Default = in any month within the 12-month window, at least one of:
- delinquency status 03 or higher (90+ days), or `RA` (REO acquisition)
- zero balance code 02, 03, 09, or 15 (credit-event terminations)

Other outcomes:
- Zero balance code 01 (voluntary payoff): non-default, loan censored; reported as a share of eligible loans (about 10% in the window at age 24).
- Zero balance code 16 or 96 ending the loan inside the window: **excluded** (non-credit terminations); count reported.
- Code 15 (whole loan sale) is a judgment call; included because the file computes an Actual Loss for it. Revisit if it matters.
- Loans that had an earlier 90+ spell that cured stay eligible (80 at age 24); only a new event in the window counts.

## Features (rule)
- Allowed: origination fields, plus performance fields **observed at or before the prediction month**.
- Never allowed: anything tagged `outcome` in `schema_map.yaml` (zero balance code and date, actual loss, recoveries, expenses, defect date, cramdown costs).

## EAD and LGD (PROPOSED)
- EAD = current actual UPB at the prediction month.
- LGD = Pooled mean 0.230 from 134 liquidated loans, floored at 0 and capped at 1. A split by origination LTV was tried and not used because the segments overlapped and went the wrong way.

## Open checks
- Share of window defaults flagged as disaster-related (`disaster_delinq_flag`) or in a borrower assistance plan (`borrower_assist_plan`): report as a sensitivity, do not silently remove.
- Exact exclusion counts for the data funnel.
