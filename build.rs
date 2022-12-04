fn main() {
    if std::env::var("CARGO_FEATURE_lapack").is_ok() {
        println!("cargo:rustc-link-lib=lapack");
    }
}
