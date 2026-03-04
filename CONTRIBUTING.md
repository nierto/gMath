# Contributing to gMath

Thank you for your interest in contributing to gMath! Contributions are welcome via GitHub issues and pull requests.

## How to Contribute

1. **Report bugs** — open a GitHub issue
2. **Suggest features** — open a GitHub issue
3. **Submit code** — fork the repo, make changes on a branch, open a pull request

All contributions are reviewed by the maintainer(s) before merging.

## Reporting Bugs

Please open an issue on GitHub with:

1. A clear title describing the problem
2. Steps to reproduce
3. Expected vs. actual output
4. Profile used (`GMATH_PROFILE=embedded|balanced|scientific`)
5. Rust version (`rustc --version`)

If reporting a precision issue, include the input value, expected result (from a reference like Wolfram Alpha or mpmath), and the actual result.

## Feature Requests

Open an issue describing:

- The use case and why it matters
- How it aligns with gMath's zero-float, multi-domain architecture
- Any design considerations or trade-offs you've identified

## Pull Request Process

1. **Fork** the repository on GitHub
2. **Create a feature branch** from `main` (`git checkout -b my-feature`)
3. Make your changes, ensuring all existing tests pass
4. Add tests for new functionality
5. Update documentation if the public API changes
6. **Push** your branch and **open a pull request** with a clear description of what changed and why

Maintainers will review and provide feedback. Please be patient — thorough review is essential for a precision-critical library.

## Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/gMath.git
cd gMath

# Select profile (default: embedded)
export GMATH_PROFILE=embedded

# Build — pre-built tables are included, takes ~2 seconds
cargo build

# Run tests
cargo test --release

# Run the comprehensive validation suite
cargo test --release --test comprehensive_benchmark -- --nocapture --test-threads=1
```

**Switching profiles**: clear the incremental build cache first to avoid stale artifacts:

```bash
rm -rf target/debug/incremental/
export GMATH_PROFILE=scientific
cargo build
```

**Rebuilding lookup tables** (optional — only if modifying table generation):

```bash
cargo build --features rebuild-tables   # ~20-30 minutes
```

## Code Standards

### Absolute Constraints

- **No f32/f64 in internal logic.** The library exists to replace floats. Float types are only permitted in `to_f64()`/`from_f64()` user-convenience methods.
- **No heap allocation on the hot path.** ZASC (Zero-Allocation Stack Computation) is a core architectural guarantee.
- **All arithmetic must be overflow-checked.** Use `checked_mul`/`checked_add` or UGOD tier promotion — never silently wrap.

### Testing

- All tests must use real computation — no mocks or placeholders.
- Precision claims must be verified against mpmath at 250+ digit reference values.
- New transcendental functions require ULP validation across the full domain.

### Style

- Follow the existing `<domain>_<operation>.rs` naming convention for domain arithmetic files.
- Keep functions focused — prefer small, well-named functions over long ones.
- Document public API items with doc comments. Internal comments should explain *why*, not *what*.

## Architecture Overview

If you're new to the codebase, start with these files:

- `src/fixed_point/canonical.rs` — the public API entry point
- `src/fixed_point/universal/zasc/lazy_expr.rs` — expression tree builder
- `src/fixed_point/universal/zasc/stack_evaluator/mod.rs` — evaluation engine
- `src/fixed_point/universal/ugod.rs` — tiered overflow delegation trait

The CLAUDE.md file contains a detailed architectural map of the entire codebase.

## License Agreement

By submitting a pull request, you agree that your contribution is licensed under the same terms as the project: [MIT OR Apache-2.0](LICENSE-MIT). You retain copyright over your contribution.
