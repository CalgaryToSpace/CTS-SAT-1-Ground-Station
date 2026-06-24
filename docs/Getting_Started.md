# Getting Started

## Set Up `uv`

1. Install the `uv` package manager: https://docs.astral.sh/uv/getting-started/installation/
2. Clone this repo, and open a terminal in the repo root.
3. Run `uv sync --dev` to create a virtual environment and install all workspace dependencies.

## Using Tools

### SatNOGS Decoder

1. Run `uv run cts1_decode_satnogs_packets --help` to confirm installation and
learn more about the tool.
2. Make an account at SatNOGS and sign in.
3. Export data frames from the "Data Export" section at: https://db.satnogs.org/satellite/GGCH-4346-1583-9419-5634#data
4. Run `uv run cts1_decode_satnogs_packets --input-csv "PATH" --output-csv "PATH"`
    * Replace the paths with your SatNOGS input data and desired output location.

### Other Mission Ops Tools

Refer to the `packages/cts1_mo_tools/README.md` file for a list of helpful local Mission Ops tools. For each tool, run `cts1_<tool_name> --help` to learn more about that tool.

## Development

1. Complete the Getting Started section above.
2. Run `uv run pytest` to execute the test suite.
3. Run `uv run ruff check .` to lint the codebase.
4. Run `uv run pyright` to run static type checking.
