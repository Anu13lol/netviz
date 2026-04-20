import math
from onnx import shape_inference

def count_params(model):
    total_param = 0

    print("\n[Metrics] Analyzing Memory Footprint...")
    print(f"{'Tensor Name':<45} | {'Shape':<20} | {'Parameters'}")
    print("-" * 60)

    for tensor in model.graph.initializer:
        shape = list(tensor.dims)

        num_elements = math.prod(shape) if shape else 1
        total_param += num_elements

        print(f"{tensor.name:<45} | {str(shape):<20} | {num_elements:,}")

    print("-" * 85)
    print(f"[Metrics] Total Parameters: {total_param:,}")

    return total_param

def flops_calc(model):
    inferred_model = shape_inference.infer_shapes(model)

    weights = {init.name: init for init in inferred_model.graph.initializer}

    activations = {val.name: val.type.tensor_type.shape for val in inferred_model.graph.value_info}
    total_flops = 0
    flops_dict = {}

    print("\n[Metrics] Analyzing Computational Cost (FLOPs)...")
    print(f"{'Node Name':<35} | {'Type':<10} | {'FLOPs'}")
    print("-" * 70)

    for node in inferred_model.graph.node:
        if node.op_type == "Conv":
            weights_name = node.input[1]
            if weights_name not in weights:
                continue

            w_shape = list(weights[weights_name].dims)
            if len(w_shape) != 4:
                continue

            c_out, c_in, kh, kw = w_shape

            output_name = node.output[0]
            if output_name not in activations:
                continue

            out_shape = activations[output_name].dim

            try:
                h = out_shape[2].dim_value
                w = out_shape[3].dim_value
            except IndexError:
                continue

            if h == 0 or w == 0:
                continue
            
            flops = 2 * c_in * c_out * kh * kw * h * w
            total_flops += flops
            flops_dict[node.name] = flops

            print(f"{node.name:<35} | {node.op_type:<10} | {c_in*c_out*kh*kw*h*w:,}")

    print("-" * 70)
    print(f"[Metrics] Total Conv FLOPs: {total_flops:,}")
    
    return total_flops, flops_dict