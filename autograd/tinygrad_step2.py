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
            print(f"new value grad= {out.grad}, data = {out.data} ")
            print(f"actual grad= {self.grad}, data = {self.data} ")
            print(f"other grad= {other.grad}, data = {other.data} \n")
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
            print("after grad opperation ------------\n")
            print(f"new value grad= {out.grad}, data = {out.data} ")
            print(f"actual grad= {self.grad}, data = {self.data} ")
            print(f"other grad= {other.grad}, data = {other.data} ")
        
        _backward()
        out._backward = _backward
    
        
        return out 

    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        
        def _backward():
            print(f"out: grad= {out.grad}, data = {out.data} ")
            print(f"actual: grad= {self.grad}, data = {self.data} ")
            print(f"other: grad= {other.grad}, data = {other.data}\n ")
          
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            print("after grad opperation ------------\n")
            print(f"new value grad= {out.grad}, data = {out.data} ")
            print(f"actual grad= {self.grad}, data = {self.data} ")
            print(f"other grad= {other.grad}, data = {other.data} ")
        
        
        out._backward = _backward
        
        return out
    
    def __pow__(self, power):
        assert isinstance(power, (int, float))
        out = Value(self.data ** power, (self,), f"**{power}" )
        
        def _backward():
            self.grad += (power * (self.data ** (power-1))) * out.grad
        
        out._backward = _backward
        return out

def tests_sum():
    w = Value(2)
    x = Value(3)
    a = x * w
    f = a * w


def tests_mul():
    w = Value(2)
    x = Value(3)
    a = x * w
    f = a * w

def tests_pow():
    w = Value(2)
    x = Value(3)
    f = w ** 2
    print(f)      

tests_mul()