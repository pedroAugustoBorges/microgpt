import os
import random
import urllib.request
import math

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
voc_size = len(uchars) + 1
print("BOS:", BOS)
print("voc_size:", voc_size)
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
        return Value (self.data * other.data, (self, other), (1,1))
    



a = Value(3.0)
b = Value(5.0)
f = Value(2.0)

c = b + a + f

print(a.data)
print(b.data)
print(c.data)
# print(c._children[0].data, c._children[1].data, c._children[2].data)  