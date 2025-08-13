# Internal Development Tools and Data

This directory contains development artifacts, profiling data, and internal tools that support the development and validation of the Parrot library.

## Contents

- **`profiling/`** - Performance benchmarking data and comparison scripts
  - Contains profiling results comparing Parrot against other libraries (CuPy, JAX, PyTorch, TensorFlow, Thrust)
  - Includes CUDA profiling data (nsys results) and benchmark implementations
  
- **`flowchart/`** - Visualization tools and dependency diagrams
  - Tools for generating library dependency visualizations
  - Generated charts showing the complete dependency structure

## Purpose

These artifacts are included in the repository to:
- Enable reproducible performance benchmarks
- Provide transparency in performance claims
- Help contributors understand performance characteristics
- Support validation of optimization efforts

## Usage

The profiling data and tools in this directory are primarily for library maintainers and contributors interested in performance analysis. They are not required for normal library usage.
