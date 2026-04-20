from rich.console import Console
from rich.table import Table
def print_arch_table(model, total_param, total_flops, flops_dict):
    console = Console()

    table = Table(title="Hardware Execution Graph", show_lines = True)

    table.add_column("Op Type", style="cyan", justify="center")
    table.add_column("Node Name", style="green", justify="center")
    table.add_column("Compute Cost (FLOPs)", style="magenta", justify="right")

    for node in model.graph.node:
        node_name = node.name if node.name else f"unnamed_{node.op_type}"

        if node.name in flops_dict:
            compute_cost = f"{flops_dict[node.name]:,}"
        else:
            compute_cost = "---"

        table.add_row(node.op_type, node_name, compute_cost)

    console.print(table)

    console.print(f"\n[bold magenta]Total Parameters (Memory):[/bold magenta] {total_param:,}")
    console.print(f"[bold magenta]Total FLOPs (Compute):[/bold magenta] {total_flops:,}\n")