# Purpose

This is a rewrite of vDUQ from scratch to ensure that I fully understand all aspects of the paper and am competent with pytorch.

vDUQ combines an inducing point GP trained with variational optization for inducing points with deep kernel learning.

This can all be launched in a docker container

## Todo

- The minimum lipshitz constant is not yet being regularized as this is done in vDUQ by using a resnet, and the resnet has not yet been implemented here
- No testing yet
- The Approximate GP is very niave and simple in every sense
- The modifications for batchnorm have not been done