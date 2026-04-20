# NetViz

A command-line tool for hardware-aware neural network analysis. Feed it an ONNX model and it parses the computation graph, calculates hardware metrics, and renders a detailed architecture diagram.

## What it does

- Parses ONNX model graphs and extracts every operation node
- Counts total trainable parameters across all layers
- Calculates FLOPs per Conv layer using exact kernel and activation dimensions
- Renders a color-coded PDF architecture diagram with directed edges
- Prints a formatted summary table in the terminal
- Exports a JSON report of all metrics

## Usage

```bash
python netviz.py path/to/model.onnx
python netviz.py path/to/model.onnx --json
```

## Output

- Terminal summary table with op type, node name, and compute cost
- `output/network_architecture.pdf` — full architecture diagram
- `output/report.json` — hardware metrics report (with `--json` flag)

## Installation

```bash
git clone https://github.com/Anu13lol/netviz.git
cd netviz
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Also install Graphviz system dependency from https://graphviz.org/download and add it to PATH.

## Supported formats

- ONNX (`.onnx`) — full support

Coming soon: Keras `.h5`, PyTorch `.pt`

## Project structure

```
netviz.py        — CLI entry point
onnxparser.py    — ONNX graph parser
metrics.py       — parameter count and FLOPs calculation
render.py        — Graphviz PDF renderer
summary.py       — rich terminal table
```