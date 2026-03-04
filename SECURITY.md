# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in gMath, please report it responsibly:

1. **Do not** open a public issue
2. Email the maintainer directly or use GitHub's private vulnerability reporting
3. Include a clear description of the vulnerability and steps to reproduce

## Scope

gMath is a deterministic arithmetic library designed for consensus-critical applications (blockchain, financial auditing). Security concerns include:

- **Precision correctness**: Any input that produces incorrect results (wrong ULP, silent overflow, precision loss without error)
- **Denial of service**: Inputs that cause panics, infinite loops, or excessive resource consumption
- **Non-determinism**: Any case where the same input produces different results across platforms or runs
- **Integer overflow**: Unchecked arithmetic that wraps silently instead of returning an error

## Response

Confirmed vulnerabilities will be:

1. Acknowledged within 72 hours
2. Investigated and patched promptly
3. Disclosed via a GitHub security advisory after a fix is available
