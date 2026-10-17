# Contributing to gMath

Thank you for your interest in contributing to gMath! Contributions are welcome via GitHub issues and pull requests.

## How to Contribute

1. **Report bugs**: open a GitHub issue
2. **Suggest features**: open a GitHub issue
3. **Submit code**: fork the repo, make changes on a branch, open a pull request

All contributions are reviewed by the maintainer(s) before merging.

## Reporting Bugs

Please open an issue on GitHub with:

1. A clear title describing the problem
2. Steps to reproduce
3. Expected vs. actual output
4. The format used (e.g. `Q64_64`, `Fixed<i32, 10>`) and the g_math version
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

Maintainers will review and provide feedback. Please be patient: thorough review is essential for a precision-critical library.

## Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/g_math.git
cd g_math

# Build: every format is compiled in every build; pre-built tables are included
cargo build

# Run the tests (four test binaries; a few minutes in release)
cargo test --release --features serde

# One test file: --test <group> <file>::
cargo test --release --test transcendentals validation_benchmark:: -- --nocapture --test-threads=1

# The 32-bit splits CI also runs (8, 12, 20, 24 fraction bits), in their own target dir
RUSTFLAGS="--cfg gmath_all_splits" CARGO_TARGET_DIR=target-splits cargo test --release --features serde

# The differential gate against the published 0.6.7 (one build per 0.6.7 profile)
cd differential && GMATH_PROFILE=realtime GMATH_FRAC_BITS=10 cargo test --release
```

The integration tests are grouped into four binaries (`tests/transcendentals.rs`,
`decimal_canonical.rs`, `linalg.rs`, `geometry.rs`); a new test file is added
there as a module. Tests that cover several formats are written once as a
`suite!` macro and instantiated with `for_each_format!` (`tests/common/mod.rs`).

**Rebuilding lookup tables** (optional, only if modifying table generation):

```bash
cargo build --features rebuild-tables   # ~20-30 minutes
```

## Code Standards

### Absolute Constraints

- **No f32/f64 in internal logic.** The library exists to replace floats. Float types are only permitted in `to_f64()`/`from_f64()` user-convenience methods.
- **No heap allocation on the hot path.** FASC (Fixed-Allocation Stack Computation) is a core architectural guarantee.
- **All arithmetic must be overflow-checked.** Use `checked_mul`/`checked_add` or UGOD tier promotion; never silently wrap.

### Testing

- All tests must use real computation: no mocks or placeholders.
- Precision claims must be verified against mpmath at 250+ digit reference values.
- New transcendental functions must be correctly rounded across the full domain and range, gated against mpmath at every format.

### Style

- Follow the existing `<domain>_<operation>.rs` naming convention for domain arithmetic files.
- Keep functions focused: prefer small, well-named functions over long ones.
- Document public API items with doc comments. Internal comments should explain *why*, not *what*.

## Architecture Overview

If you're new to the codebase, start with these files:

- `src/fixed_point/fixed_num.rs` and `width.rs`: the format type `Fixed<S, F>` and its word traits
- `src/fixed_point/canonical.rs`: the canonical entry point
- `src/fixed_point/universal/fasc/lazy_expr.rs`: expression tree builder
- `src/fixed_point/universal/fasc/stack_evaluator/mod.rs`: evaluation engine
- `src/fixed_point/exact/` and `domains/binary_fixed/transcendental/mpf.rs`: correct rounding (fast and sure stages)
- `src/fixed_point/universal/ugod.rs`: tiered overflow delegation trait

The design documents in `docs/design/` explain the larger decisions
(`WIDTH_AS_TYPE.md`, `CORRECT_ROUNDING.md`, `CERTIFIED_INTERVALS.md`).

## License Agreement

By submitting a pull request, you agree that your contribution is licensed under the same terms as the project: [MIT OR Apache-2.0](LICENSE-MIT). You retain copyright over your contribution.
