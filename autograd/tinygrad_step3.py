class Value:
    
    def __init__(self, data, _children= (), _op="" ):
        self.data = float(data)
        self.grad = 0.0
        
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda:None
        
    def __repr__(self):
        return f"Value (data={self.data:6f}, grad={self.grad:6f}, nop='{self._op}' \n children ={self._prev})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value (self.data + other.data, (self, other), "+")
        
        def _backward():
          
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        
        _backward()
        out._backward = _backward
    
        
        return out 

    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        
        def _backward():
            
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
           
        
        
        out._backward = _backward
        
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Value(self.data ** power, (self,), f"**{power}" )
        
        def _backward():
            self.grad += (power * (self.data ** (power-1))) * out.grad
        
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v, depth=0):
            indent = "  " * depth
            print(f"{indent}BUILD -> visiting node: op='{v._op}', data={v.data}, grad={v.grad}")

            if v not in visited:
                print(f"{indent}  not visited yet, adding to visited")
                visited.add(v)

                for child in v._prev:
                    print(f"{indent}  going to child: op='{child._op}', data={child.data}, grad={child.grad}")
                    build(child, depth + 1)

                topo.append(v)
                print(f"{indent}  appended to topo: op='{v._op}', data={v.data}, grad={v.grad}")
            else:
                print(f"{indent}  already visited, skipping")

        print("\n" + "=" * 60)
        print("STEP 1 -> BUILD TOPOLOGICAL ORDER")
        print("=" * 60)
        build(self)

        print("\n" + "=" * 60)
        print("STEP 2 -> TOPO ORDER (forward order)")
        print("=" * 60)
        for i, node in enumerate(topo):
            print(f"{i}: op='{node._op}', data={node.data}, grad={node.grad}")

        print("\n" + "=" * 60)
        print("STEP 3 -> SEED OUTPUT GRADIENT")
        print("=" * 60)
        self.grad = 1.0
        print(f"output node gets grad=1.0 -> op='{self._op}', data={self.data}, grad={self.grad}")

        print("\n" + "=" * 60)
        print("STEP 4 -> RUN BACKWARD IN REVERSED TOPO ORDER")
        print("=" * 60)

        for i, v in enumerate(reversed(topo)):
            print(f"\nBACKWARD STEP {i}")
            print(f"before _backward -> op='{v._op}', data={v.data}, grad={v.grad}")

            if v._prev:
                print("parents:")
                for parent in v._prev:
                    print(f"  parent op='{parent._op}', data={parent.data}, grad={parent.grad}")
            else:
                print("parents: none")

            v._backward()

            print(f"after _backward -> op='{v._op}', data={v.data}, grad={v.grad}")

            if v._prev:
                print("parents after update:")
                for parent in v._prev:
                    print(f"  parent op='{parent._op}', data={parent.data}, grad={parent.grad}")
            
def trace(root):
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def print_graph(root):
    nodes, edges = trace(root)
    print("\nNODES:")
    for n in nodes:
        print(f"  op={n._op:>4s} data={n.data:8.4f} grad={n.grad:8.4f} id={id(n)}")

    print("\nEDGES (child -> parent):")
    for (src, dst) in edges:
        print(f"  {id(src)} -> {id(dst)}    ({src._op} -> {dst._op})")


def to_dot(root, names=None):
    nodes, edges = trace(root)

    if names is None:
        names = {}

    lines = ["digraph G {", "rankdir=LR;"]

    for n in nodes:
        node_name = names.get(n, str(id(n)))

        label = f"{node_name}\\n{n._op}\\ndata={n.data:.4f}\\ngrad={n.grad:.4f}"

        lines.append(f'"{id(n)}" [label="{label}", shape="box"];')

    for src, dst in edges:
        lines.append(f'"{id(src)}" -> "{id(dst)}";')

    lines.append("}")

    return "\n".join(lines)

def print_node(name, v):
    print(f"{name:6s} data={v.data:8.4f} grad={v.grad:8.4f} op={v._op}")

if __name__ == "__main__":
    

    w = Value(2.0)
    x = Value(3.0)
    
    a = w * x
    b = a + 1
    L = b ** 2 
    
    names = {
    w: "w",
    x: "x",
    a: "a",
    b: "b",
    L: "L"
    }
    
    print("\n FORWARD:")
    print_node('w', w)
    print_node('w', x)
    print_node('a', a)
    print_node('b', b)
    print_node('L', L)
    
   
    
    L.backward()
    
    print("\nBACKWARD:")
    print_node("w", w)
    print_node("x", x)
    print_node("a", a)
    print_node("b", b)
    print_node("L", L)
    
   
    


    