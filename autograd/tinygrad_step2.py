class Value:
    
    def __init__(self, data, _children= (), _op="" ):
        self.data = float(data)
        self.grad = 2.0
        
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda:None
        
    def __repr__(self):
        return f"Value (data={self.data:6f}, grad={self.grad:6f}, op='{self._op}'. children ={self._prev})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value (self.data + other.data, (self, other), "+")
        
        def _backward():
            print(f"new value grad= {out.grad}, data = {out.data} ")
            print(f"actual grad= {self.grad}, data = {self.data} ")
            print(f"other grad= {other.grad}, data = {other.data} ")
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
            print("after grad opperation ------------")
            print(f"new value grad= {out.grad}, data = {out.data} ")
            print(f"actual grad= {self.grad}, data = {self.data} ")
            print(f"other grad= {other.grad}, data = {other.data} ")
        
        _backward()
        out._backward = _backward
    
        
        return out 
    
w = Value(2)
x = Value(3)
a = x + w
print(a)
print('-----------')
print(w, x )