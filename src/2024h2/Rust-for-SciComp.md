# TEMPLATE (replace with title of your goal)

> **Instructions:** Copy this template to a fresh file with a name based on your plan.
> Update the text. Feel free to replace any text with anything, but there are placeholders
> designed to help you get started. Also, while this template has received some iteration,
> it is not sacrosant. Feel free to change the titles of sections or make other changes that you think 
> will increase clarity.

| Metadata | |
| --- | --- |
| Owner(s) | ZuseZ4 / Manuel S. Drehwald |
| Teams | t-lang, t-compiler |
| Status | WIP |

## Motivation

Scientific Computing, High Performance Computing, and Machine Learning share the interesting challenge that they (to different degrees)
care about (very) efficient library and algorithm implementations, but are not always used by people knowledgeable in computer science.
Rust is in a nice position because Ownership, Lifetimes, and the strong Type system can prevent a descent amount of bugs. At the same
time strong alias information allows very nice performance optimizations in these fields, with performance gains well beyond what you
see in normal C++ vs. Rust Performance Comparisons. We hope to extend this by integrating Automatic Differentiation, Batching/Vectorization,
and GPU/TPU/... Offloading into the Rust language. This would put Rust on pair with the most popular features from popular Python libraries 
like JAX or PyTorch.  

### The status quo


Rust has an excellent Pyton InterOp Story thanks to PyO3. C++ has a weak interop story and as such I've seen cases where a slower C library is 
used as the backend for some Python libraries, because it was easier to bundle. Fortran is mostly used in legacy places and hardly used for new 
projects. As a solution, many researchers try to limit themself to features which are offered by compilers and libraries build on top of Python,
like JAX, PyTorch, or newly Mojo. Rust has a lot of features which make it more suitable to develop a fast and reliable backend for performance 
critical software than those languages. However, it lacks features which developers now got used to. These features are *trivial GPU usage*. 
Almost every language has some way of calling hand-written CUDA/ROCm/Sycl Kernels, but the interesting feature of languages like Julia, or libraries
like JAX is that they offer users to write Kernels in a (subset) of their already known language, without having to learn anything new. Minor 
performance penalties are not that critical in such cases, if the alternative are CPU only solution, because projects like Rust-CUDA end up being unmaintained
due to beeing too much effort to maintain outside of the LLVM or Rust project. 


*Elaborate in more detail about the problem you are trying to solve. This section is making the case for why this particular problem is worth prioritizing with project bandwidth. A strong status quo section will (a) identify the target audience and (b) give specifics about the problems they are facing today. Sometimes it may be useful to start sketching out how you think those problems will be addressed by your change, as well, though it's not necessary.*

### The next few steps

1) Merge the `#[autodiff]` fork.
2) Expose the experimental Batching feature of Enzyme, preferably by a new contributor.
3) Merge a MVP `#[offloading]` fork which is able to run simple functions using rayong parallelism on a GPU or TPU, showing a speed-up.

### The "shiny future" we are working towards

All three proposed features (batching, autodiff, offloading) can be combined and work nicely together. We have State-of-the-art support libraries like faer to cover linear
algebra and we start to see more and more libraries in other languages use Rust with these features as their backend. Cases which don't require interactive exploration also become more popular in pure Rust.

## Design axioms

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

*You can also include other resources as relevant, such as hardware, domain names, or whatever else.*

### Support needed from the project

* Discussion on CI: It would be nice to test the Offloading support on at least all 3 mayor GPU Vendors. I am somewhat confident that I can find someone to set up something, but it would be good to discuss how to maintain this best.

* Discussions on Design and Maintainability: I expect Feedback regarding usability from Users. However, I will probably keep asking questions on zulip, which might take some time (either from lang/compiler, or other teams).

## Outputs and milestones

### Outputs

An `#[Offload]` rustc-builtin-macro which makes a function definition known to the LLVM offloading backend.
A bikeshead `offload!([GPU1, GPU2, TPU1], foo(x, y,z));` macro which will execute function foo on the specified devices.
An `#[Autodiff]` rustc-builtin-macro which differentiates a given function.
A `#[Batching]` rustc-builtin-macro which fuses N function calls into one call, enabling better vectorization. 

### Milestones

*Milestones you will reach along the way*

## Frequently asked questions

### Do these changes have to happen in the compiler?
- No! Both features could be implemented in user-space, if the Rust compiler would support Reflection. In this case I could ask the compiler for the optimized backend IR for a given function. I would then need use either the AD or Offloading abilities of the LLVM library to modify the IR, generating a new function. The user would then be able to call that newly generated function. This would require some discussion on how we can have crates in the ecosystem that work with various LLVM versions, since crates are usually expected to have a MSRV, but the LLVM (and like GCC/Cranelift) backend will have breaking changes.

### Batching? 
- Offered by all autodiff tool, JAX has an extra command for it, whereas Enzyme (the autodiff backend) combines Batching with AutoDiff. We might want to split these since both have value on their own.

### What do I do with this space?

*This is a good place to elaborate on your reasoning above -- for example, why did you put the design axioms in the order that you did? It's also a good place to put the answers to any questions that come up during discussion. The expectation is that this FAQ section will grow as the goal is discussed and eventually should contain a complete summary of the points raised along the way.*