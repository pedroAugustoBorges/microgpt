import os
import random
import urllib.request
import math
from anytree import Node, RenderTree


random.seed(42)

if not os.path.exists('input.txt'):
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()] 
random.shuffle(docs)
print(f"num docs: {len(docs)}")

print('-----------------------------\n')
print("Extract all letter between each word in dataset and get the uniques letters")
"""
"""
uchars = sorted(set("".join(docs)))
print("type uchars", type(uchars))
print("len uchars", len(uchars))
print("uchars:", uchars)

print('-----------------------------\n')

BOS = len(uchars)
vocab_size = len(uchars) + 1
print("BOS:", BOS)
print("voc_size:", vocab_size)
print("possible and valid ids:", (0, len(uchars) - 1))
print("special id:", BOS)

print('-----------------------------\n')


print("Encode")

"""
Example "pedro" -> (p 15), (e 4), (d 3), (r 17) (o 14)
BOS = 26
Output is 26 + (p 15), (e 4), (d 3), (r 17) (o 14) + 26
"""


doc = docs[0]
tokens = [BOS] + [uchars.index(word) for word in 'pedro'] + [BOS]
print('pedro')
print(tokens)

print('-----------------------------\n')

print('Decode')

def decode(tokens):
    out = []
    for t in tokens:
        if t == 26:
            out.append("<BOS>")
        else:
            out.append(uchars[t])
    
    return "".join(out)


print('decoded:', decode(tokens))



"""
Autograd
"""
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    
    def __init__(self, data, children= (), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
        
    
    def __add__(self, other):
        print('>>> __add__ was called')
        other = other if  isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    
    def __mul__(self, other):
        print('>>> __mul__ was called')
        other = other if isinstance(other, Value) else Value(other)
        return Value ( (self.data * other.data), (self, other), (other.data, self.data))
    
    def __pow__(self, other):
        return Value(self.data ** other, (self, ), (other * self.data**(other-1)) )

    def log(self):
        return Value(math.log(self.data), (self,), (1/self.data))
    
    def exp(self):
        return Value(math.exp(self.data), (self), self(math.exp(self.data)))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0)))

    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * (other)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self **-1
    
    def backward(self):
        topo = []
        
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1
        
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                print(child.data, local_grad )
                child.grad += local_grad * v.grad



    
    
a = Value(3.0)

b = Value(5.0)
f = Value(2.0)

c = b + a + f
print(a.data)
print(b.data)
print(c.data)
c.backward()


#Initialize the parameters, to store the knowledge of the model

n_layer = 1
n_emb = 16
block_size = 16
n_head = 4
head_dim = n_emb // n_head
matrix = lambda nout, nin, std = 0.08 : [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte' : matrix(vocab_size, n_emb), 'wpe' : matrix(block_size, n_emb), 'lm_head' : matrix(vocab_size, n_emb)  }


for x in matrix(vocab_size, n_emb):
    for y in x:
        print(y.data)

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_emb, n_emb)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_emb, n_emb)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_emb, n_emb)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_emb, n_emb)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_emb, n_emb)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_emb, 4 * n_emb)


print(state_dict.keys())


def print_vector(name, vec):
    print(f"\n{name}")
    print("-" * 40)

    for l in vec:
        print(f' {len(l)})- {len(vec)}')
        for i, v in enumerate(l):
            print(f"layer {name} - {i:02d}: data={v.data:.4f} grad={v.grad:.4f}")
        
print_vector('attn_wq', state_dict['layer0.attn_wq'])    
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")