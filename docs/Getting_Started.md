## Getting Started

### Set Up `uv`

1. Install the `uv` package manager: https://docs.astral.sh/uv/getting-started/installation/
2. Clone this repo, and open a terminal in the repo root.
3. Run `uv sync` to create a virtual environment and install all workspace dependencies.

### Using Tools

#### SatNOGs Decoder

1. Run `uv run packages/cts1_gs_tool_lib/src/cts1_gs_tool_lib/decode_beacons.py --help` to confirm installation and
learn more about the script.
4. Run `uv run packages/cts1_gs_tool_lib/src/cts1_gs_tool_lib/decode_beacons.py --input-csv "PATH" --output-csv "PATH"`
replacing the paths with your SatNOGs input data and desired output location.

## Development

1. Complete the Getting Started section above.
2. Run `uv run pytest` to execute the test suite.
3. Run `uv run ruff check .` to lint the codebase.
4. Run `uv run pyright` to run static type checking.
