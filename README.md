# Atlas Engine
[![build](https://github.com/evroon/atlas/actions/workflows/build.yml/badge.svg)](https://github.com/evroon/atlas/actions/workflows/build.yml)
[![style](https://github.com/evroon/atlas/actions/workflows/style.yml/badge.svg)](https://github.com/evroon/atlas/actions/workflows/style.yml)

[![preview](https://raw.githubusercontent.com/evroon/atlas/master/etc/preview.png)](https://github.com/evroon/atlas/tree/master/etc/preview.png)

A Vulkan graphics engine written in Rust using [Vulkano](https://github.com/vulkano-rs/vulkano).

*Preview image is rendered using Crytek's version of the Sponza atrium model (by Marko Dabrovic)*

# Features
* Display UI with egui
* Deferred renderer
* Render 3D models using assimp

# Usage
After installing Rust, run the following commands to install additional dependencies on Ubuntu and
to fetch the Sponza model.

```bash
scripts/setup.sh
scripts/get_assets.sh
```

It's advised to build the code in release mode, because loading PNG textures in debug mode is very slow.
To run the application, run:

```bash
cargo run --release
```
