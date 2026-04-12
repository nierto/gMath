//! Domain routing — fractal topology classification and routing table.
//!
//! **PURPOSE**: Eliminate domain crisscross in arithmetic by classifying operands
//! via CompactShadow denominator factoring and dispatching through a compile-time
//! routing table in .rodata.

pub mod fractal_topology;
pub use fractal_topology::{
    DomainChoice, OperandClass, OpId,
    classify, classify_denominator, classify_literal_string,
    route_binary_op, route_unary_op, route_expression,
    coerce_to_decimal,
    TERNARY_BIT, BINARY_BIT, DECIMAL_BIT, SYMBOLIC_BIT,
};
