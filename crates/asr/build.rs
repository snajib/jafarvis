use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
  let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
  let whisper_dir = manifest_dir.join("../../vendor/whisper.cpp");

  println!(
    "cargo:rerun-if-changed={}",
    whisper_dir.to_str().unwrap()
  );

  let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

  // CMake build configuration
  let mut config = cmake::Config::new(&whisper_dir);

  config
    .define("BUILD_SHARED_LIBS", "OFF")
    .define("WHISPER_BUILD_TESTS", "OFF")
    .define("WHISPER_BUILD_EXAMPLES", "OFF")
    .define("WHISPER_BUILD_SERVER", "OFF");

  // CRITICAL: GGML_NATIVE=OFF prevents -mcpu=native flag which Apple clang rejects on ARM
  config.define("GGML_NATIVE", "OFF");

  // Metal disabled for initial implementation — enable later for GPU acceleration
  config.define("GGML_METAL", "OFF");

  config.profile("Release");

  if cfg!(target_os = "macos") {
    config.define("GGML_ACCELERATE", "ON");
    println!("cargo:rustc-link-lib=framework=Accelerate");
  }

  let dst = config.build();

  println!("cargo:rustc-link-search=native={}/lib", dst.display());
  println!("cargo:rustc-link-lib=static=whisper");
  // whisper depends on ggml libraries
  println!("cargo:rustc-link-lib=static=ggml");
  println!("cargo:rustc-link-lib=static=ggml-cpu");
  println!("cargo:rustc-link-lib=static=ggml-blas");
  println!("cargo:rustc-link-lib=static=ggml-base");

  if cfg!(target_os = "macos") {
    println!("cargo:rustc-link-lib=dylib=c++");
  } else {
    println!("cargo:rustc-link-lib=dylib=stdc++");
  }

  // Bindgen FFI generation
  let mut builder = bindgen::Builder::default()
    .header(whisper_dir.join("include/whisper.h").to_str().unwrap())
    // whisper.h includes ggml.h — tell clang where to find it
    .clang_arg(format!(
      "-I{}",
      whisper_dir.join("ggml/include").to_str().unwrap()
    ));

  // Platform-specific clang args for system headers
  if cfg!(target_os = "macos") {
    // Use xcrun to get SDK path dynamically
    if let Ok(output) = Command::new("xcrun").arg("--show-sdk-path").output() {
      if output.status.success() {
        let sdk_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        builder = builder.clang_arg(format!("-I{}/usr/include", sdk_path));
      }
    }
  }

  let bindings = builder
    .layout_tests(false)
    .allowlist_type("whisper_.*")
    .allowlist_function("whisper_.*")
    .allowlist_var("WHISPER_.*")
    .derive_debug(true)
    .derive_copy(true)
    .derive_eq(true)
    .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    .generate()
    .expect("Unable to generate whisper bindings");

  bindings
    .write_to_file(out_dir.join("whisper_bindings.rs"))
    .expect("Couldn't write bindings");
}
