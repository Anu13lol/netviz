import os
import graphviz

# Dictionary mapping op types to hex colors
COLOR_MAP = {
    'Conv': '#1f497d',               # Dark Blue
    'BatchNormalization': '#7f8c8d', # Slate Grey
    'Relu': '#1abc9c',               # Teal Green
    'Add': '#f39c12',                # Orange
    'MaxPool': '#9b59b6',            # Purple
    'GlobalAveragePool': '#9b59b6',
    'AveragePool': '#9b59b6',
    'Gemm': '#c0392b',               # Dark Red
    'MatMul': '#c0392b',
    'Linear': '#c0392b'
}

def render_graph(model):
    """Generates a visual PDF representation of the computational DAG."""
    print("\n[Renderer] Generating hardware architecture graph...")
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Left to right layout
    dot = graphviz.Digraph(name='NetViz_Arch', format='pdf')
    dot.attr(rankdir='TB', size='20, 200', dpi='150', nodesep='0.4', ranksep='0.8') 
    
    # Lookup dictionary: tensor_name -> node_name
    tensor_to_producer = {}
    
    # --- PASS 1: Build Nodes & Populate Lookup ---
    for node in model.graph.node:
        node_id = node.name if node.name else f"unnamed_{node.op_type}"
        
        # Map every output tensor this node produces to its node_id
        for out_tensor in node.output:
            tensor_to_producer[out_tensor] = node_id
            
        # Apply color map
        bg_color = COLOR_MAP.get(node.op_type, '#e0e0e0')
        # Ensure text is readable on dark backgrounds
        text_color = 'white' if bg_color in ['#1f497d', '#c0392b', '#9b59b6', '#7f8c8d'] else 'black'
        
        # Clean up node names (strip long ONNX paths if present)
        short_name = node_id.split('/')[-1] if '/' in node_id else node_id
        
        # HTML-like labels for multi-line formatting with font sizes
        label = f'<<B>{node.op_type}</B><BR/><FONT POINT-SIZE="10">{short_name}</FONT>>'
        
        dot.node(
            node_id, 
            label=label, 
            shape='box', 
            style='rounded,filled', 
            fillcolor=bg_color, 
            fontcolor=text_color,
            fontname='Helvetica'
        )

    # --- PASS 2: Route Edges ---
    for node in model.graph.node:
        node_id = node.name if node.name else f"unnamed_{node.op_type}"
        
        # For every input this node takes, find who produced it and draw the line
        for in_tensor in node.input:
            if in_tensor in tensor_to_producer:
                producer_id = tensor_to_producer[in_tensor]
                dot.edge(producer_id, node_id)

    # Export to file
    output_path = os.path.join(output_dir, "network_architecture")
    dot.render(output_path, cleanup=True) 
    
    print(f"[Renderer] Architecture graph exported successfully: {output_path}.pdf")