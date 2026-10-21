"""One timing policy for CuTe DSL and cuBLASLt, using the same operand workspace."""

import contextlib
import inspect
import math

TIMING_METHOD = "cuda_graph"
HOST_METHOD = "host_events"


@contextlib.contextmanager
def example_current_stream(module):
    """Route the pinned NVFP4 run()'s default-stream helper to the current stream.

    The example explicitly calls cutlass_torch.default_stream(), ignoring the
    surrounding PyTorch stream context. Restore the helper even if run() fails.
    No upstream file or kernel implementation is edited.
    """
    helper = getattr(module, "cutlass_torch", None)
    if helper is None:
        yield
        return
    if int(helper.current_stream()) == 0:
        raise ValueError("the example requires a non-default current stream")
    original = helper.default_stream
    helper.default_stream = helper.current_stream
    try:
        yield
    finally:
        helper.default_stream = original


class PairedTimer:
    """Graph replay is primary; a host-enqueued event series is a paired diagnostic.

    Each method gets its own warm-up. Compilation, allocations and graph capture
    precede timing. Graph support is required, never silently substituted.
    """

    def __init__(self, reverse=False):
        import cutlass.testing as testing
        self.original = testing.benchmark
        if "use_cuda_graphs" not in inspect.signature(self.original).parameters:
            raise RuntimeError("the pinned CUTLASS benchmark lacks CUDA Graph support")
        self.reverse, self.last = reverse, None

    def benchmark(self, callable, **kwargs):
        if int(kwargs.get("stream", 0)) == 0:
            raise ValueError("paired GEMM timing requires an explicit non-default stream")
        if kwargs.get("workspace_count", 1) != 1:
            raise ValueError("this study requires one hot-cache operand workspace")
        if int(kwargs.get("warmup_iterations", 0)) < 1 or int(kwargs.get("iterations", 0)) < 1:
            raise ValueError("positive warm-up and timed launch counts are required")
        kwargs = dict(kwargs)
        # Preparing a new workspace for each timing method would change operands
        # and invalidate the matched-input comparison in Experiment V.
        generator = kwargs.pop("workspace_generator", None)
        if generator is not None:
            if kwargs.get("kernel_arguments") is not None:
                raise ValueError("ambiguous operand workspace")
            kwargs["kernel_arguments"] = generator()
        order = (HOST_METHOD, TIMING_METHOD) if self.reverse else (TIMING_METHOD, HOST_METHOD)
        values = {}
        for method in order:
            value = float(self.original(callable, **{
                **kwargs, "use_cuda_graphs": method == TIMING_METHOD, "use_cupti": False}))
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError(f"invalid {method} duration")
            values[method] = value
        self.last = {"timing_method": TIMING_METHOD,
                     "host_event_time_us": values[HOST_METHOD],
                     "host_to_graph_time_ratio": values[HOST_METHOD]/values[TIMING_METHOD],
                     "timing_order": "/".join(order)}
        return values[TIMING_METHOD]

    @contextlib.contextmanager
    def intercept(self):
        """Apply the policy to unchanged pinned example run() functions."""
        import cutlass.testing as base
        from cutlass.cute import testing as cute_testing
        modules = list({id(module): module for module in (base, cute_testing)}.values())
        originals = [(module, module.benchmark) for module in modules]
        for module, _ in originals:
            module.benchmark = self.benchmark
        try:
            yield self
        finally:
            for module, original in originals:
                module.benchmark = original
