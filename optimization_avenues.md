# Possible Ways of Opitmizing the Code
- `StepwiseTangentEM::optimize` is still 24.7% of the runtime. We should be able to bring it down by a large factor.
- `Li` is 54% of the runtime.
    - `sampleSurface` is 19% of the runtime.
        - `STree.find()` takes 5.2% of the runtime. However, we are traversing the tree three times, once during sampling, and once when copying samples back into the tree, and once when finding its neighbor cell. This can be reduced by a factor of 1/3.
    - `std::shared_ptr` overhead is 3.2%. Completely avoidable.
    - `push_back_synchronized` is 9.3%! This should be better, but no obvious ways of fixing it come to mind.
