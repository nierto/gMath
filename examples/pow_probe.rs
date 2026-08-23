fn main() {
    use g_math::canonical::{gmath, evaluate};
    for (x, y, want) in [("0.5","2","0.25"), ("2","-1","0.5"), ("0.5","-2","4"), ("2","2","4"), ("1.5","2","2.25")] {
        let r = evaluate(&gmath(x).pow(gmath(y)));
        println!("pow({x},{y}) [want {want}] -> {:?}", r.map(|v| format!("{v}")));
    }
}
