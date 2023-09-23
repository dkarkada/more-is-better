# more-is-better

## notes for supercomputer software stack

* as of 15 sep 2023
* need nightly pytorch for cudnn 8.9.0.131 (bundles with cuda 12.1 support)... i think neural tangents needs this updated cudnn
* but need to install pytorch via pip not conda.. conda fails to install with gpu support (???)

## random gotchas

* jax pseudoinverse has a different tolerance than numpy pseudoinverse
* jax inverse fails silently compared to torch inverse on large kernel matrices (only on delta, not colab). I suspect it's because my delta kernels are 64-bit and jax doesn't like float64s. But i didn't test this hypothesis i just switched to torch.linalg.inv
* (i just changed all datasets and kernel tiles to float32 so hopefully the issue is moot now.)
* in general, if train error diverges with increasing n (at zero ridge), there's something going wrong with the inverse
* jax greedily self-allocated gpu memory. If i call a jitted function with 32gb free gpu mem, jax grabs it and pytorch starves. Use jax config to use cpu exclusively