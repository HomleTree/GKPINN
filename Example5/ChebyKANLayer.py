import torch
import torch.nn as nn


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y
    
# 这段代码定义了一个自定义的PyTorch模块 `ChebyKANLayer`，用于实现基于切比雪夫多项式的内核近似网络层。

# ### 初始化 (`__init__` 方法)
# ```python
# class ChebyKANLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, degree):
#         super(ChebyKANLayer, self).__init__()
#         self.inputdim = input_dim
#         self.outdim = output_dim
#         self.degree = degree

#         # 初始化切比雪夫系数矩阵
#         self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
#         nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

#         # 注册一个不可训练的张量作为范围 [0, 1, 2, ..., degree]
#         self.register_buffer("arange", torch.arange(0, degree + 1, 1))
# ```
# - `input_dim`: 输入特征的维度。
# - `output_dim`: 输出特征的维度。
# - `degree`: 切比雪夫多项式的阶数。

# 在初始化中，首先创建了一个可训练的参数 `cheby_coeffs`，它是一个形状为 `(input_dim, output_dim, degree + 1)` 的张量，用来存储切比雪夫多项式的系数。这些系数将在前向传播中与输入数据结合，以实现近似功能。

# `arange` 是一个不可训练的张量，包含了从 `0` 到 `degree` 的整数，用于在计算中乘以对应的角度。

# ### 前向传播 (`forward` 方法)
# ```python
# def forward(self, x):
#     # 将输入 x 规范化到 [-1, 1] 的范围内
#     x = torch.tanh(x)
#     # 将 x 视图重塑为 (batch_size, inputdim, 1)，然后在第三维度上复制 (degree + 1) 次
#     x = x.view((-1, self.inputdim, 1)).expand(-1, -1, self.degree + 1)
#     # 对 x 中的每个元素应用反余弦函数
#     x = x.acos()
#     # 将 x 乘以 arange 中的对应元素 [0, 1, 2, ..., degree]
#     x *= self.arange
#     # 应用余弦函数
#     x = x.cos()
#     # 计算切比雪夫插值
#     y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)
#     # 将输出 y 重塑为 (batch_size, outdim)
#     y = y.view(-1, self.outdim)
#     return y
# ```
# - 首先，使用 `torch.tanh` 将输入 `x` 规范化到 `[-1, 1]` 的范围内。
# - 将 `x` 的形状调整为 `(batch_size, inputdim, 1)`，然后在第三个维度上复制 `degree + 1` 次，这样可以创建一个形状为 `(batch_size, inputdim, degree + 1)` 的张量。
# - 对 `x` 中的每个元素应用反余弦函数 (`acos`)。
# - 将结果乘以 `arange`，即将每个角度乘以对应的 `[0, 1, 2, ..., degree]`。
# - 应用余弦函数 (`cos`)，得到切比雪夫多项式的值。
# - 使用 `torch.einsum` 函数计算张量乘积，将每个输入的切比雪夫多项式乘以对应的系数 `cheby_coeffs`，得到输出 `y`。这一步实现了切比雪夫插值。
# - 最后，将输出 `y` 的形状重塑为 `(batch_size, outdim)`，并返回作为前向传播的结果。

# 这段代码实现了一个能够对输入数据进行切比雪夫插值的自定义PyTorch层，用于构建基于切比雪夫多项式的核近似网络。