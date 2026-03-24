#!/usr/bin/env python3
"""Generate mpmath reference values for Q32.32 compact profile validation.

Outputs Rust const arrays for all 18 transcendental functions at Q32.32 precision.
Q32.32: 32 fractional bits, ~9.6 decimal digits, scale = 2^32 = 4294967296.
"""

import mpmath
mpmath.mp.dps = 50  # 50-digit precision for reference

SCALE = 2**32

def to_q32_32(val):
    """Convert mpmath value to Q32.32 raw i64."""
    scaled = val * SCALE
    return int(mpmath.nint(scaled))

# Test points: avoid boundaries, stay within Q32.32 range
test_points = {
    'exp':  ['0.5', '1.0', '-0.5', '2.0', '-1.0', '0.1'],
    'ln':   ['0.5', '1.0', '2.0', '3.0', '0.1', '10.0'],
    'sqrt': ['0.25', '1.0', '2.0', '4.0', '9.0', '0.5'],
    'sin':  ['0.0', '0.5', '1.0', '-1.0', '1.5707963', '3.14159265'],
    'cos':  ['0.0', '0.5', '1.0', '-1.0', '1.5707963', '3.14159265'],
    'tan':  ['0.0', '0.5', '1.0', '-0.5', '0.7853981', '-0.7853981'],
    'atan': ['0.0', '0.5', '1.0', '-1.0', '0.1', '100.0'],
    'asin': ['0.0', '0.5', '-0.5', '0.9', '-0.9', '0.1'],
    'acos': ['0.0', '0.5', '-0.5', '0.9', '-0.9', '0.1'],
    'sinh': ['0.0', '0.5', '1.0', '-1.0', '2.0', '-0.5'],
    'cosh': ['0.0', '0.5', '1.0', '-1.0', '2.0', '-0.5'],
    'tanh': ['0.0', '0.5', '1.0', '-1.0', '2.0', '-0.5'],
    'asinh': ['0.0', '0.5', '1.0', '-1.0', '2.0', '10.0'],
    'acosh': ['1.0', '1.5', '2.0', '3.0', '10.0', '1.1'],
    'atanh': ['0.0', '0.5', '-0.5', '0.9', '-0.9', '0.1'],
}

funcs = {
    'exp': mpmath.exp, 'ln': mpmath.log, 'sqrt': mpmath.sqrt,
    'sin': mpmath.sin, 'cos': mpmath.cos, 'tan': mpmath.tan,
    'atan': mpmath.atan, 'asin': mpmath.asin, 'acos': mpmath.acos,
    'sinh': mpmath.sinh, 'cosh': mpmath.cosh, 'tanh': mpmath.tanh,
    'asinh': mpmath.asinh, 'acosh': mpmath.acosh, 'atanh': mpmath.atanh,
}

print('//! mpmath-validated reference values for Q32.32 compact profile')
print('//! Generated with mpmath at 50-digit precision')
print('//! Format: (input_str, input_raw_i64, expected_raw_i64, function_name)')
print()
print('pub const Q32_REFS: &[(&str, i64, i64, &str)] = &[')

for fname, points in test_points.items():
    f = funcs[fname]
    for p in points:
        x = mpmath.mpf(p)
        try:
            y = f(x)
            x_raw = to_q32_32(x)
            y_raw = to_q32_32(y)
            # Verify input fits in i64
            if abs(x_raw) < 2**63 and abs(y_raw) < 2**63:
                print(f'    ("{p}", {x_raw}_i64, {y_raw}_i64, "{fname}"),')
        except Exception as e:
            print(f'    // {fname}({p}) skipped: {e}')

print('];')
