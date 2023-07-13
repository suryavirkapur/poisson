# poisson
- a poisson eqn solver written in cuda using a multi-cell method
- a cpu process requests/computations one by one, whereas a gpu can process requests parallelly
- therefore, computations that require a lot of data manipulations are better processed on a gpu
- this program uses CUDA (an nvidia framework) to parallelize soln to poisson's eqn
