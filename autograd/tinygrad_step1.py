class Value:
    def __init__(self, data):
        self.data = float(data)
        self.grad = 0.0
        
    def __repr__(self):
        return f"Value (data= {self.data:.6f}, grad={self.grad:6f})"

w = Value(2)
x = Value(3)

print(w, x )