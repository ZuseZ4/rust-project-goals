# Scientific Computing in Rust

| Metadata | |
| --- | --- |
| Owner(s) | ZuseZ4 / Manuel S. Drehwald |
| Teams | t-lang, t-compiler |
| Status | WIP |

## Motivation

Scientific Computing, High Performance Computing (HPC), and Machine Learning (ML) share several interesting challenges:
1. Highly efficient library and algorithm implementations are necessary, but are not always used by people experienced in computer science.
2. Performance-sensitive code often needs to be developed and maintained by people with more domain expertise than computer science experience.
3. Correct code must satisfy numerical approximations of continuous invariances and stability properties, which are hard to encode in types and hard to reliably unit test.
4. Algorithms make heavy use of derivatives, so productivity requires differentiable programming.
5. There is a need for "collective" parallel reasoning for multi-threaded, multi-process, and GPU code.

Rust is attractive because Ownership, Lifetimes, and the strong type system can prevent many bugs that are common to write and expensive to find. At the same
time strong alias information enables performance gains well beyond what you see in typical C++ vs. Rust performance comparisons because alias analysis due to interaction with Automatic Differentiation and GPU/Co-Processor offloading, where aliasing has both codegen and space consequences.

### The status quo

C and C++ are the prevailing implementation language for low-level libraries and performance-sensitive code. In some domains, these C/C++ parts appear in the form of extension modules for Python, which is very popular as a front-end, especially in the ML space. Fortran (in both older and modern dialects) is common for legacy HPC applications, especially in climate and weather simulation and engineering tools, but is increasingly seen as a liability for new projects and its relatively poor support for encapsulation has limited its uptake for libraries. Build systems, portability, and packaging are ever-present challenges for C, C++, and Fortran projects, as well as Python projects that incorporate extension modules.

Rust has an excellent Python interop story thanks to PyO3. C++ is more difficult to call from Python, and some projects stick to older/slower/less featureful C libraries for Python interop simplicity. The popularity of optimizing systems for restricted dialects of Python, such as JAX, PyTorch, and newly Mojo, has increased. The restrictions lead to some surprises and create a schism between these modern ML-focused tools and more general scientific computing (often using adaptive algorithms that are painful to express in the ML-focused tools). Julia embraces JIT compilation and differentiable programming at a core level, which is both a blessing (expressivity) and a curse (maintainability, static analysis).

Rust is attractive for developing performance-critical software in these domains, but lacks a robust numerical/scientific computing library ecosystem and key language support, most notably *trivial GPU usage*.
Almost every language has some way of calling hand-written CUDA/HIP/SYCL kernels (written in these C++ dialects or pre-compiled to the likes of PTX or SPIR-V), but the interesting feature of languages like Julia and libraries like JAX is that they enable users to write kernels in a (subset) of their familiar host language.
Minor performance penalties are often tolerable in exploratory or rapidly-changing code, especially when the alternative is CPU-only.
Rust-CUDA was the closest effort to this for Rust, but it is CUDA-only with its own `rustc_codegen_nvvm`, has been unmaintained for two years, and would be a heavy lift to maintain outside the Rust project or upstream.
The other major effort in this area is Rust-GPU from Embark, which is more graphics-focused and requires a pinned toolchain and its own `rustc_codegen_spirv`.

*Elaborate in more detail about the problem you are trying to solve. This section is making the case for why this particular problem is worth prioritizing with project bandwidth. A strong status quo section will (a) identify the target audience and (b) give specifics about the problems they are facing today. Sometimes it may be useful to start sketching out how you think those problems will be addressed by your change, as well, though it's not necessary.*

### The next few steps

1) Merge the `#[autodiff]` fork.
2) Expose the experimental batching feature of Enzyme, preferably by a new contributor.
3) Merge a MVP `#[offloading]` fork which is able to run simple functions using Rayon parallelism on a GPU or TPU, showing a speed-up.
4) Complete `ptx-kernel` ABI ([tracking issue](https://github.com/rust-lang/rust/issues/38788)) and expose shared memory, shuffles, etc.
5) Extend bitcode-linker (`LinkerFlavor::Llbc`) and related tooling to support AMDGPU and SPIR-V targets.
6) Ergonomics for target-agnostic kernels and device functions, likely including Cargo `multidep` ([tracking issue](https://github.com/rust-lang/cargo/issues/10030)).
7) Contribute to `std::simd` and the [struct target features RFC](https://github.com/rust-lang/rfcs/pull/3525).
8) Explore safety for kernels (preferring library-based approaches that abstract away handling the raw unsafe kernel).
9) This project is interested in collaborating with [Contracts and Invariants](Contracts-and-invariants.md), and would benefit from `generic_const_exprs` and `adt_const_params` (which are valuable for [dimensional analysis](https://github.com/Tehforsch/diman/) and similar invariant modeling). It would also benefit from [Seamless C Support](Seamless-C-Support.md).

### The "shiny future" we are working towards

Rust becomes a the premier language for reliable, high-performance numerical libraries and applications. A rich library ecosystem develops by a combination of incremental porting and fresh Rust libraries (exemplified by [faer](https://docs.rs/faer/) for linear algebra).

Static analysis tools and advanced testing methods become widespread, increasing the reliability of HPC workflows, including distributed memory and GPU-intensive workloads. In rapidly-changing projects, new production runs more consistently succeed and produce meaningful results, reducing the need for debugging at scale.

Programmers find they are more productive in Rust due to the tooling, stronger type system, dimensional analysis, and well-integrated autodiff, batching, and offloading. Maintainers find they can more confidently review contributions and spend less effort on portability quirks and build systems. While number of users increases, their community support burden declines because the compiler and integrated documentation helps users get it right before needing to ask. Younger contributors are more interested in sticking around an becoming maintainers because the work is more interesting, accessible, and enjoyable.

Rust becomes more popular as a backend in interactive environments like Python, and resulting models can be more reliably deployed to production/edge environments as pure-Rust builds.

## Design axioms


### Offloading
- We try to provide a safe, simple and opaque offloading interface. 
- The "unit" of offloading is a function. 
- We try to not expose explicit data movement if ownership gives us enough information.
- Users can offload functions that contain parallel CPU code, but do not have final control over how the paralelism will be translated to co-processors.
- We accept that hand-written CUDA/ROCm/.. kernels might be faster, but actively try to reduce differences.
- We accept that we might need to provide additional control to the user to guide parallelism, if performance differences remain unacceptable large.
- Offloaded code might not return exact same values as code executed on the CPU. We will work with t-(opsem?) to develop clear rules.

### Autodiff
- We try to provide a fast autodiff interface that supports most autodiff features relevant for Scientific Computing.
- The "unit" of autodiff is a function.
- We acknowledge our responability since user-implemented autodiff without compiler knowledge might struggle to cover gaps in our features.
- We have a fast solution ("git plumbing") with further optimization opportunities, but need to improve safety and usability ("git porcelain").
- We need to teach users more about AutoDiff "pitfalls" and provide guides on how to handle them. [https://arxiv.org/abs/2305.07546](paper).
- We do not support differentiating (inline) assembly. Users are expected to write "custom derivatives" in such cases.
- We might refuse to expose certain features if they are too hard to use correctly and provide little gains (e.g. derivatives with respect to global vars).

### GPU kernels
- Decouple semantics of correctness (arch-independent) from arch-specific performance optimizations.
- It should be possible to test "device" functions on the host.

*Add your [design axioms][da] here. Design axioms clarify the constraints and tradeoffs you will use as you do your design work. These are most important for project goals where the route to the solution has significant ambiguity (e.g., designing a language feature or an API), as they communicate to your reader how you plan to approach the problem. If this goal is more aimed at implementation, then design axioms are less important. [Read more about design axioms][da].*

[da]: ../about/design_axioms.md

## Ownership and other resources

**Owner:** ZuseZ4 / Manuel S. Drehwald

Manuel S. Drehwald working 5 days/wk, sponsored by LLNL and the University of Toronto (UofT).
He has a background in HPC and worked on a rust compiler fork, as well as an LLVM based autodiff tool 
for the last 3 years during his undergrad. He is now in a research based Master Program. 
Supervision and Discussion on the LLVM side with Johannes Doerfert and Tom Scogland.

Resources:
Domain and CI for the autodiff work provided by MIT. Might be moved to the LLVM org later this year.
Hardware for Benchmarks provided by LLNL and UofT.
CI for the offloading work provided by LLNL or LLVM(?, see below).

### Support needed from the project

* Discussion on CI: It would be nice to test the Offloading support on at least all 3 mayor GPU Vendors. I am somewhat confident that I can find someone to set up something, but it would be good to discuss how to maintain this best in the longer term.

* Discussions on Design and Maintainability: I will probably keep asking questions to achieve a nice internal Design on zulip, which might take some time (either from lang/compiler, or other teams).

## Outputs and milestones

### Outputs

An `#[offload]` rustc-builtin-macro which makes a function definition known to the LLVM offloading backend.
A bikeshead `offload!([GPU1, GPU2, TPU1], foo(x, y,z));` macro which will execute function foo on the specified devices.
An `#[autodiff]` rustc-builtin-macro which differentiates a given function.
A `#[batching]` rustc-builtin-macro which fuses N function calls into one call, enabling better vectorization. 

### Milestones

- The first offloading step is the automatic copying of a slice or vector of floats to a device and back.

- The second offloading step is the automatic translation of a (default) `Clone` implementation to create a Host2Device and Device2Host copy implementation for user types.

- The third offloading step is to run some embarassingly parallel Rust code (e.g. scalar times Vector) on the GPU. 

- Fourth we have examples of how rayon code runs faster on a co-processor using offloading.

- Stretch-goal. Combining Autodiff and Offloading in one example that runs differentiated code on a GPU.

## Frequently asked questions

### Why do you implement these features only on the LLVM Backend? 
- Performance-wise we have LLVM and GCC as performant backends. Modularity wise we have LLVM and especially Cranelift being nice to modify.
  It seems resonable that LLVM thus is the first backend to have support for new features in this field. Especially the offloading support 
  should be supportable by other compiler backends, given pre-existing work like OpenMP offloading and WebGPU.

### Do these changes have to happen in the compiler?
- No! Both features could be implemented in user-space, if the Rust compiler would support reflection. In this case I could ask the compiler for the optimized backend IR for a given function. I would then need use either the AD or offloading abilities of the LLVM library to modify the IR, generating a new function. The user would then be able to call that newly generated function. This would require some discussion on how we can have crates in the ecosystem that work with various LLVM versions, since crates are usually expected to have a MSRV, but the LLVM (and like GCC/Cranelift) backend will have breaking changes, unlike Rust.

### Batching? 
- Offered by all autodiff tools, JAX has an extra command for it, whereas Enzyme (the autodiff backend) combines Batching with AutoDiff. We might want to split these since both have value on their own. Some libraries also offer Array-of-Struct vs Struct-of-Array features which are related but often have limited usability or performance when implemented in userspace. To be fair this is a less mature feature of Enzyme, so I could understand concerns. However, following the Autodiff work this feature can be exposed in very few (100?) loc. My main bieksheding thoughts where about whether we want to pass 3 batched args as [x1, x2, x3], (x1, x2, x3), or x1, x2, x3. Also it's a nice feature to get something started, once the main autodiff PR got merged.

### Writing a GPU Backend in 6 months sounds tough..
- True. But similar to the Autodiff work I'm exposing something that's already existing in the Backend. I just don't think that Rust, Julia, C++, Carbon, Fortran, Chappel, Haskell, Bend, Python, ... should all write their own GPU or Autodiff Backends. Most of these already share compiler optimization through LLVM or GCC, so let's also share this. Of course, we should still push to use our Rust specific magic.

### Rust Specific Magic?
TODO:

### How about Safety?
I want all these features to be safe by default, and I am happy to not expose some features if the gain is too small for the safety risk.
As an Example, Enzyme can compute the derivative with respect to a global. Too niche, discouraged (and unsafe) for Rust. `¯\_(ツ)_/¯`

How to parallelize your 3 nested for loops efficiently has been researched for decades. Lately there also has been some more work on how to translate different parallelism type efficiently, e.g. from GPUs to CPUs, or now maybe some rayon parllelism to GPUs? I am therefore not particualrily worried about Correctness. 

### What do I do with this space?

*This is a good place to elaborate on your reasoning above -- for example, why did you put the design axioms in the order that you did? It's also a good place to put the answers to any questions that come up during discussion. The expectation is that this FAQ section will grow as the goal is discussed and eventually should contain a complete summary of the points raised along the way.*
