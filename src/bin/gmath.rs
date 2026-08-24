//! gmath: deterministic fixed-point calculator over the g_math crate.
//!
//! This is the REAL g_math FASC (canonical fixed-point) engine exposed as a
//! CLI: zero-float, reproducible arithmetic with exact rationals and
//! transcendentals. The same inputs produce the same result on every machine,
//! which is the whole point of the ecosystem. It replaces the former bash/`bc`
//! demo, which had nothing to do with g_math.
//!
//! Build:  cargo build --release --bin gmath
//! Usage:  gmath <op> <args...>   (run `gmath help`)

use std::env;
use std::process;

use g_math::canonical::{evaluate, gmath_parse, LazyExpr};

const USAGE: &str = "\
gmath — deterministic fixed-point calculator (g_math FASC engine)

USAGE:
    gmath <op> <args...>

BINARY OPS:
    add | +          A B    A + B
    sub | -          A B    A - B
    mul | multiply   A B    A * B
    div | divide     A B    A / B
    pow | ^          A B    A raised to B

UNARY OPS:
    sqrt             A      square root
    exp              A      e^A
    ln               A      natural logarithm
    sin | cos | tan  A      trigonometric (radians)
    atan             A      arctangent
    neg              A      -A

OTHER:
    eval             EXPR   parse + evaluate a single value
    version                 print the g_math crate version
    help                    show this message

Numbers may be integers (42), decimals (1.5), or EXACT rationals (1/3).
All arithmetic is fixed-point and deterministic: no floating-point drift,
identical results on every machine.
";

fn die(msg: String) -> ! {
    eprintln!("gmath: {msg}");
    process::exit(2);
}

fn parse(s: &str) -> LazyExpr {
    gmath_parse(s).unwrap_or_else(|e| die(format!("cannot parse '{s}': {e:?}")))
}

fn emit(expr: LazyExpr) -> ! {
    match evaluate(&expr) {
        Ok(v) => {
            println!("{v}");
            process::exit(0);
        }
        Err(e) => die(format!("evaluation failed (overflow): {e:?}")),
    }
}

fn need(args: &[String], n: usize, op: &str) {
    if args.len() != n {
        die(format!("{op}: expected {n} argument(s), got {}", args.len()));
    }
}

fn main() {
    let argv: Vec<String> = env::args().skip(1).collect();
    if argv.is_empty() {
        eprint!("{USAGE}");
        process::exit(2);
    }

    let op = argv[0].as_str();
    let rest = &argv[1..];

    match op {
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            process::exit(0);
        }
        "version" | "-V" | "--version" => {
            println!("g_math {}", g_math::VERSION);
            process::exit(0);
        }
        "add" | "+" => {
            need(rest, 2, op);
            emit(parse(&rest[0]) + parse(&rest[1]));
        }
        "sub" | "-" => {
            need(rest, 2, op);
            emit(parse(&rest[0]) - parse(&rest[1]));
        }
        "mul" | "multiply" => {
            need(rest, 2, op);
            emit(parse(&rest[0]) * parse(&rest[1]));
        }
        "div" | "divide" => {
            need(rest, 2, op);
            emit(parse(&rest[0]) / parse(&rest[1]));
        }
        "pow" | "^" => {
            need(rest, 2, op);
            emit(parse(&rest[0]).pow(parse(&rest[1])));
        }
        "sqrt" => {
            need(rest, 1, op);
            emit(parse(&rest[0]).sqrt());
        }
        "exp" => {
            need(rest, 1, op);
            emit(parse(&rest[0]).exp());
        }
        "ln" => {
            need(rest, 1, op);
            emit(parse(&rest[0]).ln());
        }
        "sin" => {
            need(rest, 1, op);
            emit(parse(&rest[0]).sin());
        }
        "cos" => {
            need(rest, 1, op);
            emit(parse(&rest[0]).cos());
        }
        "tan" => {
            need(rest, 1, op);
            emit(parse(&rest[0]).tan());
        }
        "atan" => {
            need(rest, 1, op);
            emit(parse(&rest[0]).atan());
        }
        "neg" => {
            need(rest, 1, op);
            emit(parse("0") - parse(&rest[0]));
        }
        "eval" => {
            need(rest, 1, op);
            emit(parse(&rest[0]));
        }
        other => die(format!("unknown op '{other}' (try `gmath help`)")),
    }
}
