use bindgen::builder;
use cmake;
use miette::IntoDiagnostic;
use std::env;
use std::path::PathBuf;

fn main() -> miette::Result<()> {
    let llama_path = std::env::current_dir().unwrap().join("./extern/llama-cpp");

    // Build Llama.cpp
    let dst = cmake::Config::new(&llama_path)
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("BUILD_SHARED_LIBS", "ON")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");

    // Generate bindings for Llama.cpp API
    let bindings = builder().header("wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .clang_args(["-x", "c++", "-std=c++17"])
        .clang_arg(format!("-I{}", llama_path.join("ggml/include").to_str().unwrap()))
        .generate()
        .into_diagnostic()?;

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs"))
        .into_diagnostic()?;

    Ok(())
}
