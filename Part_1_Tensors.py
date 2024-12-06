import torch
import numpy as np

# Initializing a tensor - Directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# Initializing a tensor - From a NumPy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# Initializing a tensor - From another tensor
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype = torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# Initializing a tensor - With random or constant values
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

# Attributes of a tensor
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Operations on Tensors - GPU
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# Operations on Tensors - Standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# Operations on Tensors - Joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim = 1)
print(t1)

t2 = torch.stack([tensor, tensor, tensor], dim = 1)
print(t2)

# Operations on Tensors - Arithmetic operations
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out = y3)
print(y1)
print(y3)

# This computes the element-wise product. z1, x2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out = z3)
print(z3)

# Operations on Tensors - Single-element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# Operations on Tensors - In-place operations
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Bridge with NumPy - Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Bridge with Numpy - NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

print(f"t: {t}")
print(f"n: {n}")

np.add(n, 1, out = n)

print(f"t: {t}")
print(f"n: {n}")