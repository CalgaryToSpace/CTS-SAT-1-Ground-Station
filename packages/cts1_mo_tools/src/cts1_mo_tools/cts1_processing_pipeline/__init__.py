"""CTS-SAT-1 processing pipeline.

This file exists to cap the thread pools of the libraries below it, and has
to do so here rather than at an entry point: polars sizes its thread pool
exactly once, from `POLARS_MAX_THREADS`, when it is first imported, and
ignores the variable afterwards. Python imports a package's `__init__`
before any of its submodules, so this is the one place in the package that
is guaranteed to run before the first `import polars` no matter which
module (`cli`, `web_ui.main`, a test) is the one that pulls it in.

See `resource_limits` for what's being capped and why.
"""

from cts1_mo_tools.cts1_processing_pipeline import resource_limits

resource_limits.apply_thread_limits(resource_limits.DEFAULT_POLARS_THREADS)
