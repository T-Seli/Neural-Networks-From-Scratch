import math
import random

class Value:
  def __init__(self, data, _children=(), _op=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op

  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    return out

  def __radd__(self, other):
    return self + other

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
    return out

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    return self * other**-1

  def exp(self):
    out = Value(math.exp(self.data), (self,), 'exp')
    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward
    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float)), "only supports int/float powers"
    out = Value(self.data**other, (self,), f'**{other}')
    def _backward():
      self.grad += (other * self.data ** (other - 1)) * out.grad
    out._backward = _backward
    return out

  def tanh(self):
    x = self.data
    out = Value((math.exp(2*x) - 1)/(math.exp(2*x) + 1), (self,), 'tanh')
    def _backward():
      self.grad += (1 - out.data**2) * out.grad
    out._backward = _backward
    return out

  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)

    self.grad = 1.0
    build_topo(self)
    for node in reversed(topo):
      node._backward()

class Neuron:
  def __init__(self, nin, nonlin=True, w_init=None, b_init=0.0):
    self.w = [Value(w_init() if w_init else random.uniform(-0.1, 0.1)) for _ in range(nin)]
    self.b = Value(b_init)
    self.nonlin = nonlin

  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)    # w * x + b
    return act.tanh() if self.nonlin else act

  def parameters(self):
    return self.w + [self.b]

class Layer:
  def __init__(self, nin, nout, nonlin=True):
    limit = math.sqrt(6.0 / (nin + nout))
    w_init = lambda: random.uniform(-limit, limit)
    b_init = 0.0
    self.neurons = [Neuron(nin, nonlin=nonlin, w_init=w_init, b_init=b_init) for _ in range(nout)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs

  def parameters(self):
    return [param for neuron in self.neurons for param in neuron.parameters()]

class MLP:
  def __init__(self, nin, nouts): # nounts is a list of the size of each of the layers in the MLP
    sz = [nin] + nouts
    self.layers = []
    for i in range(len(nouts)):
        is_last = (i == len(nouts) - 1)
        self.layers.append(Layer(sz[i], sz[i+1], nonlin=not is_last))

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [param for layer in self.layers for param in layer.parameters()]