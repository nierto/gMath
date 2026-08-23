fn main() {
    use g_math::canonical::{gmath, evaluate};
    println!("cos(0.1)={}", evaluate(&gmath("0.1").cos()).unwrap());
}
