pub mod mesh;
pub mod parallel;
pub mod physics;
pub mod solver;
pub mod vertical;
pub mod transport;
pub mod compute;

#[cfg(all(feature="native-metal",target_os="macos"))]
mod native_metal;

pub mod era5;

pub mod initialization;
