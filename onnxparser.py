import onnx

def load_model(file_path):
    print(f"[Parser] Loading model: {file_path}")
    model = onnx.load(file_path)
    
    try:
        onnx.checker.check_model(model)
        print(f"[Parser] Model loaded: {file_path}")
        return model
    except onnx.checker.ValidationError as e:
        print(f"[Parser] Error loading model: {file_path}")
        print(e)
        return None
    
    return model

def nodes_count(model):
    count = 0
    for n in model.graph.node:
        count += 1
    print(f"[Parser] Total nodes mapped: {count}")
    
    return count

'''
def node_type(model):
    types = []
    name = []
    
    print("\n[Parser] Extraction Complete. Node Execution Order:")
    print(f"{'Operation Type':<20} | {'Node Name'}")
    print("-" * 60)

    for n in model.graph.node:
        types.append(n.op_type)
        name.append(n.name)
    
    print(f"{n.op_type:<20} | {n.name}")
    
    return types, name
'''
def extract_nodes_type(model):
    node_data = []

    print(f"\n{'Operation Type':<20} | {'Node Name'}")
    print("-" * 60)

    for n in model.graph.node:
        node_info = {
            "type": n.op_type,
            "name": n.name
        }
        node_data.append(node_info)

        print(f"{n.op_type:<20} | {n.name}")

    return node_data