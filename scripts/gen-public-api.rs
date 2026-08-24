//! gen-public-api.rs: public-API index generator for the `g_math` crate.
//!
//! WHAT THIS IS
//! ------------
//! A single self-contained Rust program (std library only, no external crates,
//! no `cargo` deps) that scans a hardcoded, curated set of "surface" source
//! files and emits `PUBLIC_API.md` at the repo root (one entry per public
//! symbol with a one-line summary) so the README never hand-copies signatures.
//!
//! This is a PRAGMATIC SOURCE SCAN, not a compiler. It reads text with a small
//! brace-depth state machine; it does NOT type-check, expand macros, or resolve
//! re-exports. Treat its output as a high-signal index, not ground truth.
//!
//! WHAT IT SCANS
//! -------------
//! Only the files listed in `SURFACE` below (see that table). Each file maps to
//! a display module path and one output section. When a NEW public module is
//! added to the crate's reachable surface, a contributor must add a line to
//! `SURFACE` (and, if it is a new logical area, wire it into `SECTIONS`).
//!
//! WHAT IT EXTRACTS (per file)
//! ---------------------------
//! * Free items declared `pub`: `pub fn` / `pub struct` / `pub enum` /
//!   `pub trait` / `pub const` / `pub type`. Summary = the first non-empty
//!   `///` doc line immediately preceding the item (leading `/// ` stripped).
//! * Methods: `impl` blocks are tracked by brace depth (handles `impl<...> T`,
//!   `impl T`, `impl Trait for T`, and multi-line signatures). Inside an
//!   INHERENT impl, `pub fn` (plus `pub const` / `pub type` associated items)
//!   are collected. Inside a NON-std TRAIT impl, ALL `fn` are collected (they
//!   are public through the trait). Methods are attributed to their type.
//!
//! KNOWN LIMITATIONS / OMISSIONS
//! -----------------------------
//! * `pub use` RE-EXPORTS are listed in a per-section "Re-exports" table by NAME
//!   and source module (a file that is mostly re-exports, e.g. `canonical.rs`,
//!   surfaces its `gmath`/`evaluate`/`LazyExpr`/… this way). Signatures for
//!   re-exported names live on docs.rs; a re-export whose name is already scanned
//!   directly in the same section (e.g. `PlanarTQ19`) is not double-listed.
//! * Items inside `#[cfg(test)]` modules or `mod tests { .. }` are skipped.
//! * `#[doc(hidden)]` items are skipped (a count is printed to stderr).
//! * IMPL blocks for STD / common derive traits (Add, Sub, Mul, Div, Neg, Rem,
//!   *Assign, Index, Display, Debug, Clone, Copy, PartialEq, Eq, PartialOrd,
//!   Ord, Hash, From, Into, TryFrom, Default, Deref, Iterator, FromStr, Error,
//!   serde Serialize/Deserialize, ...) ARE OMITTED to avoid boilerplate noise.
//! * Brace counting is literal-aware but does not fully parse block comments
//!   spanning lines; the curated surface files avoid those patterns.
//! * `#[cfg(feature = "...")]` gates (item-level or whole-file) are recorded and
//!   shown so the reader knows what is behind a feature.
//!
//! HOW TO REGENERATE
//! -----------------
//!   rustc -O scripts/gen-public-api.rs -o /tmp/gen-public-api && /tmp/gen-public-api
//!
//! Run from the repo root (paths in `SURFACE` are repo-root relative).

use std::collections::HashSet;
use std::fmt::Write as _;
use std::fs;
use std::process;

// ============================================================================
// Curated public surface. (file, display_module, section_key, feature_gate)
// Add a line here when a new public module joins the reachable surface.
// ============================================================================
const SURFACE: &[(&str, &str, &str, Option<&str>)] = &[
    ("src/fixed_point/canonical.rs", "g_math::canonical", "canonical", None),
    ("src/fixed_point/imperative/fixed_point.rs", "FixedPoint (re-exported at g_math::fixed_point)", "fixedpoint", None),
    ("src/fixed_point/imperative/fixed_vector.rs", "FixedVector", "fixedvector", None),
    ("src/fixed_point/imperative/fixed_matrix.rs", "FixedMatrix", "fixedmatrix", None),
    ("src/fixed_point/domains/decimal_fixed/decimal_fixed.rs", "DecimalFixed", "decimal", None),
    ("src/fixed_point/imperative/fused.rs", "g_math::fixed_point::imperative::fused", "fused", None),
    ("src/fixed_point/imperative/decompose.rs", "g_math::fixed_point::imperative::decompose", "linalg", None),
    ("src/fixed_point/imperative/derived.rs", "g_math::fixed_point::imperative::derived", "linalg", None),
    ("src/fixed_point/imperative/matrix_functions.rs", "g_math::fixed_point::imperative::matrix_functions", "linalg", None),
    ("src/fixed_point/imperative/manifold.rs", "g_math::fixed_point::imperative::manifold", "geometry", None),
    ("src/fixed_point/imperative/lie_group.rs", "g_math::fixed_point::imperative::lie_group", "geometry", None),
    ("src/fixed_point/imperative/curvature.rs", "g_math::fixed_point::imperative::curvature", "geometry", None),
    ("src/fixed_point/imperative/projective.rs", "g_math::fixed_point::imperative::projective", "geometry", None),
    ("src/fixed_point/imperative/fiber_bundle.rs", "g_math::fixed_point::imperative::fiber_bundle", "geometry", None),
    ("src/fixed_point/imperative/ode.rs", "g_math::fixed_point::imperative::ode", "ode", None),
    ("src/fixed_point/imperative/tensor.rs", "g_math::fixed_point::imperative::tensor", "tensors", None),
    ("src/fixed_point/imperative/tensor_decompose.rs", "g_math::fixed_point::imperative::tensor_decompose", "tensors", None),
    ("src/fixed_point/imperative/serialization.rs", "g_math::fixed_point::imperative::serialization", "serialization", None),
    ("src/fixed_point/tq19/mod.rs", "g_math::tq19", "tq19", Some("inference")),
    ("src/fixed_point/tq19/ops.rs", "g_math::tq19", "tq19", Some("inference")),
    ("src/fixed_point/tq19/planar.rs", "g_math::tq19", "tq19", Some("inference")),
    ("src/fixed_point/tq19/hybrid.rs", "g_math::tq19", "tq19", Some("inference")),
    ("src/fixed_point/compute_tier.rs", "g_math::compute_tier", "compute_tier", Some("inference")),
    ("src/fixed_point/domains/balanced_ternary/mod.rs", "g_math::fixed_point::domains::balanced_ternary", "ternary", None),
    ("src/fixed_point/domains/balanced_ternary/trit_packing.rs", "g_math::fixed_point::domains::balanced_ternary", "ternary", None),
];

// Section rendering order + headings. (section_key, heading)
const SECTIONS: &[(&str, &str)] = &[
    ("canonical", "Canonical (g_math::canonical)"),
    ("fixedpoint", "FixedPoint"),
    ("fixedvector", "FixedVector"),
    ("fixedmatrix", "FixedMatrix"),
    ("decimal", "DecimalFixed"),
    ("fused", "Fused operations"),
    ("linalg", "Linear algebra"),
    ("geometry", "Geometry"),
    ("ode", "ODE"),
    ("tensors", "Tensors"),
    ("ternary", "Balanced ternary"),
    ("tq19", "TQ1.9 inference"),
    ("compute_tier", "Compute-tier transcendentals"),
    ("serialization", "Serialization"),
];

// Std / common derive / serde traits whose impls are omitted as boilerplate.
const STD_TRAITS: &[&str] = &[
    "Add", "Sub", "Mul", "Div", "Neg", "Rem", "AddAssign", "SubAssign",
    "MulAssign", "DivAssign", "RemAssign", "Index", "IndexMut", "Display",
    "Debug", "Clone", "Copy", "PartialEq", "Eq", "PartialOrd", "Ord", "Hash",
    "From", "Into", "TryFrom", "TryInto", "Default", "Deref", "DerefMut",
    "Iterator", "IntoIterator", "FromIterator", "AsRef", "AsMut", "Borrow",
    "BorrowMut", "Sum", "Product", "Not", "BitAnd", "BitOr", "BitXor", "Shl",
    "Shr", "ShlAssign", "ShrAssign", "Error", "Drop", "FromStr", "Send", "Sync",
    "Serialize", "Deserialize",
];

#[derive(Clone)]
struct FreeItem {
    section: String,
    name: String,
    kind: String, // "fn" | "struct" | "enum" | "trait" | "const" | "type"
    summary: String,
    feature: Option<String>,
}

#[derive(Clone)]
struct Method {
    section: String,
    type_name: String,
    name: String,
    summary: String,
}

#[derive(Clone)]
struct ReExport {
    section: String,
    name: String,
    source: String, // base module path the name is re-exported from
}

#[derive(Clone, Copy, PartialEq)]
enum ImplKind {
    Inherent,
    Trait,
    StdTrait,
}

struct ImplCtx {
    type_name: String,
    kind: ImplKind,
    close_depth: i32,
}

fn main() {
    let mut free_items: Vec<FreeItem> = Vec::new();
    let mut methods: Vec<Method> = Vec::new();
    let mut reexports: Vec<ReExport> = Vec::new();
    let mut hidden_count: usize = 0;
    let mut missing: Vec<String> = Vec::new();

    for (path, _display, section, feature) in SURFACE {
        let src = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => {
                missing.push((*path).to_string());
                continue;
            }
        };
        scan_file(
            &src,
            section,
            feature.map(|s| s.to_string()),
            &mut free_items,
            &mut methods,
            &mut reexports,
            &mut hidden_count,
        );
    }

    let md = render(&free_items, &methods, &reexports);
    if let Err(e) = fs::write("PUBLIC_API.md", &md) {
        eprintln!("error: could not write PUBLIC_API.md: {e}");
        process::exit(1);
    }

    let total = dedup_free(&free_items).len() + dedup_methods(&methods).len();
    let reex = dedup_reexports(&reexports).len();
    eprintln!("gen-public-api: wrote PUBLIC_API.md");
    eprintln!("gen-public-api: {total} public symbols emitted");
    eprintln!("gen-public-api: {reex} re-export name(s) captured");
    eprintln!("gen-public-api: {hidden_count} #[doc(hidden)] item(s) skipped");
    for m in &missing {
        eprintln!("gen-public-api: WARNING surface file not found: {m}");
    }
}

// ============================================================================
// Scanner
// ============================================================================
fn scan_file(
    src: &str,
    section: &str,
    file_feature: Option<String>,
    free_items: &mut Vec<FreeItem>,
    methods: &mut Vec<Method>,
    reexports: &mut Vec<ReExport>,
    hidden_count: &mut usize,
) {
    let mut depth: i32 = 0;
    let mut skip_depth: Option<i32> = None; // inside a test module
    let mut impl_stack: Vec<ImplCtx> = Vec::new();
    let mut doc: Vec<String> = Vec::new();
    let mut doc_hidden = false;
    let mut cfg_feature: Option<String> = None;
    let mut cfg_test = false;
    let mut impl_buf: Option<String> = None;
    let mut use_buf: Option<String> = None;

    for raw in src.lines() {
        let t = raw.trim_start();

        // --- comments & docs (no brace counting) ---
        if t.starts_with("//!") {
            continue;
        }
        if let Some(rest) = t.strip_prefix("///") {
            doc.push(rest.trim_start().trim_end().to_string());
            continue;
        }
        if t.starts_with("//") {
            continue;
        }
        if t.is_empty() {
            doc.clear();
            doc_hidden = false;
            cfg_feature = None;
            cfg_test = false;
            continue;
        }
        // --- attributes ---
        if t.starts_with("#[") || t.starts_with("#!") {
            if t.contains("doc(hidden)") {
                doc_hidden = true;
            }
            if t.contains("cfg(test)") {
                cfg_test = true;
            }
            if let Some(f) = parse_cfg_feature(t) {
                cfg_feature = Some(f);
            }
            continue;
        }

        // --- multi-line `pub use` re-export accumulation (top level only) ---
        if use_buf.is_some() || (t.starts_with("pub use ") && skip_depth.is_none()) {
            let mut buf = use_buf.take().unwrap_or_default();
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(t);
            depth += code_braces(t); // net 0 across the full statement, keeps depth balanced
            if !buf.contains(';') {
                use_buf = Some(buf);
                continue;
            }
            parse_reexports(&buf, section, reexports);
            reset(&mut doc, &mut doc_hidden, &mut cfg_feature, &mut cfg_test);
            continue;
        }

        // --- multi-line impl accumulation ---
        if impl_buf.is_some() || t.starts_with("impl ") || t.starts_with("impl<") || t == "impl" {
            let mut buf = impl_buf.take().unwrap_or_default();
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(t);
            if !buf.contains('{') {
                impl_buf = Some(buf);
                continue;
            }
            let (type_name, kind) = parse_impl(&buf);
            impl_stack.push(ImplCtx {
                type_name,
                kind,
                close_depth: depth,
            });
            depth += code_braces(t);
            pop_impls(&mut impl_stack, depth);
            reset(&mut doc, &mut doc_hidden, &mut cfg_feature, &mut cfg_test);
            continue;
        }

        // --- inside a test module: skip everything, just track depth ---
        if let Some(sd) = skip_depth {
            depth += code_braces(t);
            if depth <= sd {
                skip_depth = None;
            }
            continue;
        }

        // --- detect a test module to skip ---
        if (t.starts_with("mod ") || t.starts_with("pub mod "))
            && (cfg_test || t.contains("mod tests"))
        {
            let start = depth;
            depth += code_braces(t);
            if depth > start {
                skip_depth = Some(start);
            }
            reset(&mut doc, &mut doc_hidden, &mut cfg_feature, &mut cfg_test);
            continue;
        }

        let summary = first_nonempty(&doc);
        let feat = cfg_feature.clone().or_else(|| file_feature.clone());

        // --- methods inside an impl block ---
        if let Some(top) = impl_stack.last() {
            if depth == top.close_depth + 1 {
                // immediate impl body
                if top.kind != ImplKind::StdTrait {
                    let found: Option<String> = match top.kind {
                        ImplKind::Trait => extract_fn_name(t),
                        ImplKind::Inherent => {
                            if let Some(rest) = t.strip_prefix("pub ") {
                                let rest = rest.trim_start();
                                if let Some(n) = extract_fn_name(rest) {
                                    Some(n)
                                } else if let Some(n) = kw_ident(rest, "const") {
                                    Some(n)
                                } else if let Some(n) = kw_ident(rest, "type") {
                                    Some(n)
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                        ImplKind::StdTrait => None,
                    };
                    if let Some(n) = found {
                        if doc_hidden {
                            *hidden_count += 1;
                        } else {
                            let _ = &feat;
                            methods.push(Method {
                                section: section.to_string(),
                                type_name: top.type_name.clone(),
                                name: n,
                                summary,
                            });
                        }
                    }
                }
                depth += code_braces(t);
                pop_impls(&mut impl_stack, depth);
                reset(&mut doc, &mut doc_hidden, &mut cfg_feature, &mut cfg_test);
                continue;
            } else {
                // deeper inside a method body / nested scope
                depth += code_braces(t);
                pop_impls(&mut impl_stack, depth);
                reset(&mut doc, &mut doc_hidden, &mut cfg_feature, &mut cfg_test);
                continue;
            }
        }

        // --- free item at top level ---
        if depth == 0 {
            if let Some(rest) = t.strip_prefix("pub ") {
                let rest = rest.trim_start();
                if let Some((name, kind)) = parse_free_item(rest) {
                    if doc_hidden {
                        *hidden_count += 1;
                    } else {
                        free_items.push(FreeItem {
                            section: section.to_string(),
                            name,
                            kind,
                            summary,
                            feature: feat,
                        });
                    }
                }
            }
        }
        depth += code_braces(t);
        pop_impls(&mut impl_stack, depth);
        reset(&mut doc, &mut doc_hidden, &mut cfg_feature, &mut cfg_test);
    }
}

fn reset(
    doc: &mut Vec<String>,
    doc_hidden: &mut bool,
    cfg_feature: &mut Option<String>,
    cfg_test: &mut bool,
) {
    doc.clear();
    *doc_hidden = false;
    *cfg_feature = None;
    *cfg_test = false;
}

fn pop_impls(stack: &mut Vec<ImplCtx>, depth: i32) {
    while let Some(top) = stack.last() {
        if depth <= top.close_depth {
            stack.pop();
        } else {
            break;
        }
    }
}

fn first_nonempty(doc: &[String]) -> String {
    for l in doc {
        if !l.trim().is_empty() {
            return l.trim().to_string();
        }
    }
    String::new()
}

// Count net brace delta, ignoring braces in string/char literals and line comments.
fn code_braces(s: &str) -> i32 {
    let b = s.as_bytes();
    let n = b.len();
    let mut i = 0;
    let mut d = 0i32;
    let mut in_str = false;
    let mut in_chr = false;
    while i < n {
        let c = b[i];
        if in_str {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if in_chr {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == b'\'' {
                in_chr = false;
            }
            i += 1;
            continue;
        }
        if c == b'/' && i + 1 < n && (b[i + 1] == b'/' || b[i + 1] == b'*') {
            break; // line/block comment: stop (pragmatic)
        }
        if c == b'"' {
            in_str = true;
            i += 1;
            continue;
        }
        if c == b'\'' {
            // char literal vs lifetime: only treat as char if it looks like 'x' or '\..'
            if i + 1 < n && (b[i + 1] == b'\\' || (i + 2 < n && b[i + 2] == b'\'')) {
                in_chr = true;
                i += 1;
                continue;
            }
            i += 1;
            continue;
        }
        if c == b'{' {
            d += 1;
        } else if c == b'}' {
            d -= 1;
        }
        i += 1;
    }
    d
}

fn parse_cfg_feature(t: &str) -> Option<String> {
    // matches #[cfg(feature = "name")] or nested cfg(all(feature = "name", ...))
    let key = "feature";
    let idx = t.find(key)?;
    let after = &t[idx + key.len()..];
    let q1 = after.find('"')?;
    let rest = &after[q1 + 1..];
    let q2 = rest.find('"')?;
    Some(rest[..q2].to_string())
}

// Parse an `impl ... {` header. Returns (type_name, kind).
fn parse_impl(buf: &str) -> (String, ImplKind) {
    let head = match buf.find('{') {
        Some(i) => &buf[..i],
        None => buf,
    };
    let mut s = head.trim();
    s = s.strip_prefix("impl").unwrap_or(s).trim_start();
    // skip impl-level generics <...>
    if s.starts_with('<') {
        let mut depth = 0i32;
        let bytes = s.as_bytes();
        let mut end = 0;
        for (i, &c) in bytes.iter().enumerate() {
            if c == b'<' {
                depth += 1;
            } else if c == b'>' {
                depth -= 1;
                if depth == 0 {
                    end = i + 1;
                    break;
                }
            }
        }
        s = s[end..].trim_start();
    }
    // strip a leading `where` guard remnant is unlikely here; handle " for "
    if let Some(pos) = find_word_pos(s, "for") {
        let trait_part = s[..pos].trim();
        let type_part = s[pos + 3..].trim();
        let trait_name = last_segment(&first_type_token(trait_part));
        let type_name = last_segment(&first_type_token(type_part));
        let kind = if STD_TRAITS.contains(&trait_name.as_str()) {
            ImplKind::StdTrait
        } else {
            ImplKind::Trait
        };
        (type_name, kind)
    } else {
        (last_segment(&first_type_token(s)), ImplKind::Inherent)
    }
}

// Take the first "type token": up to whitespace / '<' / '{' / 'where'.
fn first_type_token(s: &str) -> String {
    let s = s.trim();
    let mut out = String::new();
    for ch in s.chars() {
        if ch.is_whitespace() || ch == '<' || ch == '{' || ch == '(' {
            break;
        }
        out.push(ch);
    }
    // drop a trailing "where" if it slipped in via no space (rare)
    out
}

fn last_segment(s: &str) -> String {
    let s = s.trim().trim_start_matches('&').trim();
    let s = s.strip_prefix("mut ").unwrap_or(s);
    match s.rsplit("::").next() {
        Some(x) => x.to_string(),
        None => s.to_string(),
    }
}

// Find a standalone word position (surrounded by non-identifier chars).
fn find_word_pos(s: &str, word: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let wb = word.as_bytes();
    let mut i = 0;
    while i + wb.len() <= bytes.len() {
        if &bytes[i..i + wb.len()] == wb {
            let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after_idx = i + wb.len();
            let after_ok = after_idx >= bytes.len() || !is_ident_byte(bytes[after_idx]);
            if before_ok && after_ok {
                return Some(i);
            }
        }
        i += 1;
    }
    None
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

// Extract fn name if the line declares a fn (handles const/unsafe/async fn).
fn extract_fn_name(line: &str) -> Option<String> {
    let pos = find_word_pos(line, "fn")?;
    let after = line[pos + 2..].trim_start();
    let name = read_ident(after);
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

// If `rest` starts with keyword, return the following identifier (for const/type).
fn kw_ident(rest: &str, kw: &str) -> Option<String> {
    let after = rest.strip_prefix(kw)?;
    if !after.starts_with(|c: char| c.is_whitespace()) {
        return None;
    }
    let after = after.trim_start();
    let name = read_ident(after);
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

fn read_ident(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            break;
        }
    }
    out
}

// Parse a free `pub` item body (text after "pub "). Returns (name, kind).
fn parse_free_item(rest: &str) -> Option<(String, String)> {
    // skip non-item pub forms
    if rest.starts_with("use ")
        || rest.starts_with("mod ")
        || rest.starts_with("static ")
        || rest.starts_with("extern ")
        || rest.starts_with("crate ")
    {
        return None;
    }
    // function first (covers `const fn`, `unsafe fn`, `async fn`)
    if let Some(n) = extract_fn_name(rest) {
        return Some((n, "fn".to_string()));
    }
    for (kw, kind) in [
        ("struct", "struct"),
        ("enum", "enum"),
        ("trait", "trait"),
        ("const", "const"),
        ("type", "type"),
    ] {
        if let Some(n) = kw_ident(rest, kw) {
            return Some((n, kind.to_string()));
        }
    }
    None
}

// Parse a complete `pub use <path>{::{ ... }|::Name};` statement into re-exports.
fn parse_reexports(buf: &str, section: &str, out: &mut Vec<ReExport>) {
    let s = buf.trim();
    let s = s.strip_prefix("pub use ").unwrap_or(s);
    let s = s.trim().trim_end_matches(';').trim();

    if let Some(open) = s.find('{') {
        let base = s[..open].trim().trim_end_matches(':').trim().to_string();
        let close = s.rfind('}').unwrap_or(s.len());
        let inner = &s[open + 1..close.min(s.len())];
        for part in inner.split(',') {
            if let Some(name) = reexport_name(part) {
                out.push(ReExport { section: section.to_string(), name, source: base.clone() });
            }
        }
    } else if let Some(name) = reexport_name(s) {
        // `pub use path::Name`: source is the path minus the final segment.
        let source = match s.rfind("::") {
            Some(i) => s[..i].to_string(),
            None => String::new(),
        };
        out.push(ReExport { section: section.to_string(), name, source });
    }
}

// Extract the exported identifier from one re-export entry, honoring `as` aliases
// and skipping globs / self / empty entries.
fn reexport_name(part: &str) -> Option<String> {
    let p = part.trim();
    if p.is_empty() || p == "*" || p == "self" || p.starts_with("self ") {
        return None;
    }
    // `X as Y` → exported name is Y
    if let Some(pos) = find_word_pos(p, "as") {
        let alias = p[pos + 2..].trim();
        let name = read_ident(alias);
        return if name.is_empty() { None } else { Some(name) };
    }
    // otherwise take the last path segment's identifier
    let seg = p.rsplit("::").next().unwrap_or(p).trim();
    let name = read_ident(seg);
    if name.is_empty() || name == "self" {
        None
    } else {
        Some(name)
    }
}

// ============================================================================
// Dedup
// ============================================================================
fn dedup_free(items: &[FreeItem]) -> Vec<FreeItem> {
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut out = Vec::new();
    for it in items {
        let key = (it.section.clone(), it.name.clone(), it.kind.clone());
        if seen.insert(key) {
            out.push(it.clone());
        }
    }
    out
}

fn dedup_methods(items: &[Method]) -> Vec<Method> {
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    let mut out = Vec::new();
    for it in items {
        let key = (it.section.clone(), it.type_name.clone(), it.name.clone());
        if seen.insert(key) {
            out.push(it.clone());
        }
    }
    out
}

fn dedup_reexports(items: &[ReExport]) -> Vec<ReExport> {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut out = Vec::new();
    for it in items {
        let key = (it.section.clone(), it.name.clone());
        if seen.insert(key) {
            out.push(it.clone());
        }
    }
    out
}

// ============================================================================
// Rendering
// ============================================================================
fn render(free_items: &[FreeItem], methods: &[Method], reexports: &[ReExport]) -> String {
    let free = dedup_free(free_items);
    let meth = dedup_methods(methods);
    let reex = dedup_reexports(reexports);

    let mut out = String::new();
    out.push_str("# Public API\n\n");
    out.push_str(
        "Generated from source by `scripts/gen-public-api.rs`. Do not edit by hand. \
Regenerate with: `rustc -O scripts/gen-public-api.rs -o /tmp/gen-public-api && /tmp/gen-public-api`\n\n",
    );
    out.push_str(
        "This is a **pragmatic source scan** of a curated set of surface files, not a \
compiler-verified export list. It lists `pub` free items and impl methods with the \
first line of their doc comment. `pub use` re-exports, `#[cfg(test)]` items, and \
`#[doc(hidden)]` items are omitted, as are impls of std/derive traits. See the header \
of `scripts/gen-public-api.rs` for exact scope and limitations.\n\n",
    );

    for (skey, heading) in SECTIONS {
        let sec_free: Vec<&FreeItem> = free.iter().filter(|f| f.section == *skey).collect();
        let sec_meth: Vec<&Method> = meth.iter().filter(|m| m.section == *skey).collect();
        let sec_reex: Vec<&ReExport> = reex.iter().filter(|r| r.section == *skey).collect();
        if sec_free.is_empty() && sec_meth.is_empty() && sec_reex.is_empty() {
            continue;
        }

        writeln!(out, "## {heading}\n").ok();

        // Module display paths + feature note for this section.
        let mut modules: Vec<String> = Vec::new();
        let mut feature: Option<String> = None;
        for (path, display, sk, feat) in SURFACE {
            if *sk == *skey {
                let _ = path;
                if !modules.contains(&display.to_string()) {
                    modules.push(display.to_string());
                }
                if let Some(f) = feat {
                    feature = Some(f.to_string());
                }
            }
        }
        if !modules.is_empty() {
            write!(out, "Modules: {}", modules.join(", ")).ok();
            if let Some(f) = &feature {
                write!(out, " _(feature: {f})_").ok();
            }
            out.push_str("\n\n");
        }

        // Types that have methods (rendered as ### subheadings).
        let mut method_types: Vec<String> = Vec::new();
        for m in &sec_meth {
            if !method_types.contains(&m.type_name) {
                method_types.push(m.type_name.clone());
            }
        }

        // Free items that are NOT a method-bearing type declaration → items table.
        let standalone: Vec<&FreeItem> = sec_free
            .iter()
            .filter(|f| !method_types.contains(&f.name))
            .copied()
            .collect();

        if !standalone.is_empty() {
            out.push_str("| Item | Kind | Summary |\n");
            out.push_str("| --- | --- | --- |\n");
            for f in &standalone {
                let mut sum = escape_cell(&f.summary);
                if let Some(ft) = &f.feature {
                    if feature.as_deref() != Some(ft.as_str()) {
                        sum = format!("{sum} _(feature: {ft})_").trim().to_string();
                    }
                }
                writeln!(out, "| `{}` | {} | {} |", f.name, f.kind, sum).ok();
            }
            out.push('\n');
        }

        // Re-exports whose name is not already scanned directly in this section.
        let mut direct_names: HashSet<&str> = HashSet::new();
        for f in &sec_free {
            direct_names.insert(f.name.as_str());
        }
        for m in &sec_meth {
            direct_names.insert(m.type_name.as_str());
        }
        let sec_reex_new: Vec<&ReExport> = sec_reex
            .iter()
            .filter(|r| !direct_names.contains(r.name.as_str()))
            .copied()
            .collect();
        if !sec_reex_new.is_empty() {
            out.push_str("**Re-exports**, signatures on [docs.rs](https://docs.rs/g_math):\n\n");
            out.push_str("| Item | Re-exported from |\n");
            out.push_str("| --- | --- |\n");
            for r in &sec_reex_new {
                let src = if r.source.is_empty() { "-".to_string() } else { format!("`{}`", r.source) };
                writeln!(out, "| `{}` | {} |", r.name, src).ok();
            }
            out.push('\n');
        }

        // Method-bearing types as subheadings.
        for tname in &method_types {
            writeln!(out, "### {tname}\n").ok();
            // type's own doc summary if declared in section
            if let Some(decl) = sec_free.iter().find(|f| &f.name == tname) {
                if !decl.summary.trim().is_empty() {
                    writeln!(out, "{}\n", decl.summary).ok();
                }
            }
            out.push_str("| Method | Summary |\n");
            out.push_str("| --- | --- |\n");
            for m in sec_meth.iter().filter(|m| &m.type_name == tname) {
                let sum = escape_cell(&m.summary);
                writeln!(out, "| `{}` | {} |", m.name, sum).ok();
            }
            out.push('\n');
        }
    }

    out
}

fn escape_cell(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ").trim().to_string()
}
