import torch

n, c, h, w = 1, 1, 3, 3  # batch size, channels, height, width
f, k = 1, 2  # number of filters, kernel size

x = torch.randint(-3, 3, (n, c, h, w)).float()
x.requires_grad_(True)

w = torch.randint(-3, 3, (f, c, k, k)).float()
w.requires_grad_(True)

b = torch.randint(-3, 3, (f,)).float()
b.requires_grad_(True)

out = torch.conv2d(x, w, bias=b)
print("out: ", out)

# Random upstream gradient
dout = torch.randint_like(out, -3, 3)

out.backward(dout)

print(f'x: {x}\n\nw: {w}\n\nb: {b}\n\ndout: {dout}\n')
# Given x, w, b, and dout, what's dw, db, and dx?
print(f'dw: {w.grad}\n\ndb: {b.grad}\n\ndx: {x.grad}\n')


# Given x and w, compute the transposed convolution. Note this is NOT a continuation of the
# previous example, it's just using the existing x and w values to demonstrate the calculation.
transposed_conv = torch.conv_transpose2d(x, w)
print(f'transposed convolution: {transposed_conv}')
