import os
import argparse
import json

from onnxparser import load_model
from onnxparser import nodes_count
from onnxparser import extract_nodes_type
from metrics import count_params, flops_calc
from render import render_graph
from summary import print_arch_table
def main():
    parser = argparse.ArgumentParser(description='NetViz CLI: Hardware-aware neural network analyzer.')
    parser.add_argument(
        "model_path",
        type = str,
        help="Path to the compiled model file (.onnx)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Export hardware metrics to output/report.json"
    )

    args = parser.parse_args()
    target_file = args.model_path

    if not os.path.exists(target_file):
        print(f"{target_file} does not exist")
        return
    
    model_graph = load_model(target_file)
    if model_graph is None:
        return
    
    print(f"[NetViz] Target model registered: {target_file}")

  
    #n_count = nodes_count(model_graph)
    #n_type = extract_nodes_type(model_graph)

    total_p = count_params(model_graph)
    total_flops, flops_dict = flops_calc(model_graph)

    print_arch_table(model_graph, total_param=total_p, total_flops=total_flops, flops_dict=flops_dict)

    render_graph(model_graph)

    if args.json:
        export_data = {
            "model_file": os.path.basename(target_file),
            "total_params": total_p,
            "total_flops": total_flops,
            "node_flops": flops_dict
        }

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "report.json")

        with open(report_path, "w") as f:
            json.dump(export_data, f, indent=4)

        print(f"\n[Export] Hardware metrics exported to {report_path}")
        

if __name__ == "__main__":
    main()
    