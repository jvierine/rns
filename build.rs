fn main() {
    if let Ok(root)=std::env::var("HDF5_DIR"){println!("cargo:rustc-link-arg=-Wl,-rpath,{root}/lib");}

    if std::env::var_os("CARGO_FEATURE_NATIVE_METAL").is_none() || std::env::var("CARGO_CFG_TARGET_OS").unwrap() != "macos" {return;}
    println!("cargo:rerun-if-changed=src/metal/bridge.mm");
    let out=std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let obj=out.join("metal_bridge.o");let lib=out.join("librns_metal.a");
    assert!(std::process::Command::new("clang++").args(["-O3","-fobjc-arc","-std=c++17","-c","src/metal/bridge.mm","-o"]).arg(&obj).status().unwrap().success());
    assert!(std::process::Command::new("ar").arg("rcs").arg(&lib).arg(&obj).status().unwrap().success());
    println!("cargo:rustc-link-search=native={}",out.display());println!("cargo:rustc-link-lib=static=rns_metal");
    println!("cargo:rustc-link-lib=c++");println!("cargo:rustc-link-lib=framework=Metal");println!("cargo:rustc-link-lib=framework=Foundation");
}
