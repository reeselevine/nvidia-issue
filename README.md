## Summary

This code reproduces a violation of data races being bound in time on NVIDIA GPUs (so far a GeForce RTX 4070 and a Quadro RTX 4000). The specific test is taken from the paper [_Bounding Data Races in Space and Time_](https://dl.acm.org/doi/10.1145/3192366.3192421), Section 2.2, by Dolan et al. In C++, it looks like this:

*Thread 0*
```
a = 1;
atomic_store_explicit(&flag, 1, memory_order_release);
```

*Thread 1*
```
a = 2;
while (atomic_load_explicit(&flag, memory_order_acquire) == 0);
b = a;
c = a;
```

The atomic operations serve to synchronize the two threads. A time-bound violation occurs if the data race on `a` before the synchronization leads to `b` and `c` observing different values of `a`.

Note that violations are _not_ a bug; the data race on `a` technically makes this program undefined. However, it is still interesting to see whether current compiler optimizations lead to the violation. So far, only NVIDIA GPUs seem susceptible (tested on NVIDIA, AMD, Intel, Arm).

## Test Format

We run the test using `device` memory, using `acquire/release` fences with a device scope. Threads on either side of the test may or may not be part of the same workgroup, but the majority are not in the same workgroup. Unsurprisingly, the violation does not reproduce in a simple, two-thread test. Therefore, this program utilizes techniques from our research on memory model litmus testing to increase stress on the system. Specifically, observing the violation seems to require two techniques:

1.) A parallel test strategy, as described in [MC Mutants](https://dl.acm.org/doi/10.1145/3575693.3575750), with 15-20% of dispatched workgroups dedicated to testing.

2.) Workgroup shuffling, as described in [Foundations of empirical memory consistency testing](https://dl.acm.org/doi/10.1145/3428294), where the 80-85% of workgroups that are not testing simply exit without executing any instructions.

## Building and Running

After compiling (`make`), the build folder contains an executable (`runner`) that will run the shader code and report the results. If multiple GPUs are available, one can be chosen with the `-d` flag (e.g. `./runner -d 1`). `-v` enable validation layers.

The output shows the number of times each outcome was seen on a given run. At the end, the total number of violations of the time-bound property is shown. Parameters such as the number of iterations are hardcoded in the `runner.cpp` file.
