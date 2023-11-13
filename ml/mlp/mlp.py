import numpy as np

class Value:

    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data<0 else self.data, (self,))

        def _backward():
            self.grad += (out.data>0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # first put nodes in the topological order
        # eg first all children then the node itself

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build_topo(c)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __radd__(self, other): # other + self
        return self + other

    def __rmul__(self, other): # other * self
        return self * other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __neg__(self): # -self
        return self * -1

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_in, activation=True):
        self.w = [Value(np.random.normal() * 2**0.5 / n_in**0.5) for _ in range(n_in)]
        self.b = Value(0)
        self.activation = activation

    def __call__(self, x):
        act = sum((iw*ix for iw,ix in zip(self.w, x)), self.b)
        return act.relu() if self.activation else act

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):

    def __init__(self, n_in, n_out, **kwargs):
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]

        return out[0] if len(out)==1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, n_in, n_outs):
        sizes = [n_in] + n_outs
        self.layers = [Layer(sizes[i], sizes[i+1], activation=i!=len(n_outs)-1) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


