import torch

# When training neural networks, the most frequent used algorithm is back propagation

x = torch.ones(5) # input tensor
y = torch.zeros(3) # expected output
w = torch.randn(5, 3, requires_grad = True)
b = torch.randn(3, requires_grad = True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Computing Gradients

loss.backward()
print(w.grad)
print(b.grad)

# We can only perform gradient calculations using backward once on a given graph, for performance reasons.
# If we need to do several backward calls on the same graph, we need to pass retain_graph = True to the backward call.

# Disabling Gradient Tracking

# Tracking computational history and support computation can be disabled when we have trained the model and just
# want to apply it to some input data (we only want to do forward computations through the network)

z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b

print(z.requires_grad)

# Another way to achieve the same result is to use the detach() method on the tensor

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

# There are some reasons you might want to disable gradient tracking including (1) to mark some parameters in your
# neural network as frozen parameters and (2) to speed up computations when you are only doing forward pass, because
# computations on tensors that do not track gradients would be more efficient

# More on Computational Graphs

# Conceptually, autograd keeps a record of data (tensors) and all executed operations in a directed acyclic graph (DAG)
# consisting of function objects. In this DAG, leaves are the input tensors, roots are the output sensors. By tracing
# from roots to leaves, you can compute the gradient using the chain rule.

# In a forward pass, autograd does two things simultaneously (1) run the requested operation to compute the resulting
# tensor and (2) maintain the operation's gradient function in the DAG

# THe backward pass kicks off when .backward() is called on the DAG root. autograd then (1) computes the gradients
# from each .grad_fn, (2) accumulates them in the respective tensor's .grad attribute, and (3) using the chain rule,
# propogates all the way to the leaf tensors

# DAGs are dynamic in PyTorch. The graph is recreated from scratch; after each .backward() call, autograd starts
# populating each new graph. This is what allows you to use control flow statements in your model; you can change
# the shape, size, and operation at every iteration if needed.abs

# Operational Reading: Tensor Gradients and Jacobian Products

# There are some calses when the output function is an arbitrary tensor. PyTorch allows you to compute Jacobian
# product, and not the actual gradient.

# The Jacobian matrix is basically the gradient of y and x from y1-yn and x1-xn such that the same gradient of y
# is used for each row, and the same gradient of x is used in each column.abs

# PyTorch allows you to compute Jacobian Product V^T * J for a given input vector v = (v1 ... vm). This is achieved
# by calling backward with v as an argument. The size of v should be the same as the size of the original tensor,
# with respect to which we want to compute the product.abs

inp = torch.eye(4, 5, requires_grad = True)
out = (inp + 1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph = True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph = True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph = True)
print(f"Call after zeroing gradients\n{inp.grad}")

# Quick note for me and not the guide: It looks like function calls that have a training _ modify the object that
# they are called on where ones without return a copy of the object to be assigned to another object.

# PyTorch accumulates the gradients each time backward is called. If you want to comput the proper gradients, you
# need to zero out the grad property before. In real-life training an optimizer helps us do this.

# Before we were calling backward() function without parameters. This is essentially equivalent to calling 
# backward(torch.tensor(1.0)), which is a useful way to compute the gradients in case of a scalar-valued function,
# such as loss during neural network training.