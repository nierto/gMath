#!/usr/bin/env python3
"""
Compute high-precision rational approximations via continued fractions.
Used to generate hardcoded constants for build.rs.
"""

from decimal import Decimal, getcontext
from math import sqrt, log, e, pi

# Set ultra-high precision for computation
getcontext().prec = 150

def continued_fraction(value, max_terms=50):
    """Compute continued fraction representation of a number."""
    cf = []
    x = Decimal(str(value))

    for _ in range(max_terms):
        floor_x = int(x)
        cf.append(floor_x)

        frac = x - floor_x
        if abs(frac) < Decimal('1e-100'):
            break

        x = Decimal('1') / frac

    return cf

def convergents_from_cf(cf):
    """Compute convergents from continued fraction representation."""
    convergents = []

    if len(cf) == 0:
        return convergents

    # Initialize with first two terms
    h_prev2, h_prev1 = 0, 1
    k_prev2, k_prev1 = 1, 0

    for a in cf:
        h = a * h_prev1 + h_prev2
        k = a * k_prev1 + k_prev2

        convergents.append((h, k))

        h_prev2, h_prev1 = h_prev1, h
        k_prev2, k_prev1 = k_prev1, k

    return convergents

def find_convergent_for_precision(value, target_decimals):
    """Find the convergent that gives at least target_decimals of precision."""
    cf = continued_fraction(value, max_terms=100)
    convergents = convergents_from_cf(cf)

    value_decimal = Decimal(str(value))

    for i, (num, den) in enumerate(convergents):
        if den == 0:
            continue

        approx = Decimal(num) / Decimal(den)
        error = abs(approx - value_decimal)

        # Check if error is small enough for target precision
        if error < Decimal(10) ** (-target_decimals - 1):
            return (num, den, i, float(error))

    # Return the last convergent if we can't meet precision
    return convergents[-1] + (len(convergents)-1, 0.0)

def sqrt_5_decimal():
    """Compute sqrt(5) to high precision using Decimal."""
    # Use Newton's method for square root
    x = Decimal('2.236')
    for _ in range(50):
        x = (x + Decimal('5') / x) / 2
    return x

# Mathematical constants (high precision strings)
CONSTANTS = {
    'PI': Decimal('3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679'),
    'E': Decimal('2.7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274'),
    'SQRT_2': Decimal('1.4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276415727'),
    'LN_2': Decimal('0.6931471805599453094172321214581765680755001343602552541206800094933936219696947156058633269964186875'),
    'PHI': Decimal('1.6180339887498948482045868343656381177203091798057628621354486227052604628189024497072072041893911374'),
    'LN_10': Decimal('2.3025850929940456840179914546843642076011014886287729760333279009675726096773524802359972050895982983'),
    'SQRT_3': Decimal('1.7320508075688772935274463415058723669428052538103806280558069794519330169088000370811461867572485756'),
    'SQRT_5': sqrt_5_decimal(),
}

print("=" * 80)
print("HIGH-PRECISION RATIONAL APPROXIMATIONS")
print("Continued Fraction Convergents")
print("=" * 80)
print()

# For each constant, compute approximations at 3 precision levels
for name, value in CONSTANTS.items():
    print(f"\n{name} = {value}")
    print("-" * 80)

    for precision in [19, 38, 77]:
        num, den, idx, error = find_convergent_for_precision(value, precision)
        print(f"  {precision:2d} decimals: {num:45d} / {den:45d}")
        print(f"               (convergent #{idx}, error ~{error:.2e})")

    # Also show string literal (77 decimals)
    value_str = str(value)[:80]  # First 80 chars
    print(f"  String (77d): \"{value_str}\"")

print()
print("=" * 80)
print("DENOMINATOR UNIQUENESS CHECK")
print("=" * 80)

for precision in [19, 38, 77]:
    print(f"\n{precision} decimal precision:")
    denominators = []
    for name, value in CONSTANTS.items():
        num, den, idx, error = find_convergent_for_precision(value, precision)
        denominators.append((name, den))
        print(f"  {name:8s}: {den}")

    unique_dens = len(set(d for _, d in denominators))
    total_dens = len(denominators)

    if unique_dens == total_dens:
        print(f"  ✅ PASS: All {total_dens} denominators are UNIQUE")
    else:
        print(f"  ❌ FAIL: Only {unique_dens}/{total_dens} denominators are unique!")

print()
print("Script complete. Copy values to build.rs as hardcoded constants.")
