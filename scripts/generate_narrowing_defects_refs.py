#!/usr/bin/env python3
"""mpmath references for tests/narrowing_defects_validation.rs.

Prints each value as the scaled integer the test compares against (rounded
to nearest at the stated number of decimals), 80 working digits.
"""
from mpmath import mp, mpf, exp, sqrt, nint

mp.dps = 80


def scaled(x, decimals):
    return int(nint(x * mpf(10) ** decimals))


# decimal_exp_small_is_unchanged: e^3 at the largest storage dp per profile
for d in (4, 9, 19):
    print(f"exp(3) at {d} dp: {scaled(exp(3), d)}")

# decimal_exp_in_range_is_unchanged: e^30 and e^40 at 19 dp
for x in (30, 40):
    print(f"exp({x}) at 19 dp: {scaled(exp(x), 19)}")

# realtime_decimal_sqrt_of_wide_raw: sqrt(20) at the realtime compute dp (9)
print(f"sqrt(20) at 9 dp: {scaled(sqrt(20), 9)}")
