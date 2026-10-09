fn main() {
    if let Ok(root) = std::env::var("HDF5_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{root}/lib");
    }
}
