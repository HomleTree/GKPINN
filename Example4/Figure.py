#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import matplotlib.pyplot as plt

nPt = 1001
x = np.linspace(0, 1, nPt).reshape(-1, 1)
y = np.linspace(0, 1, nPt).reshape(-1, 1)
X, Y = np.meshgrid(x, y)

data = np.loadtxt("4—0.001.csv", delimiter=",", skiprows=1)  # 跳过标题行
u = data[:, 2].flatten()[:, None]
u_star = u

u_pred_pinn = np.loadtxt("u_pred_pinn.csv", delimiter=",", skiprows=0) 
u_pred_pinn = u_pred_pinn.reshape(-1, 1)
u_pred_gkpinn = np.loadtxt("u_pred_gkpinn.csv", delimiter=",", skiprows=0) 
u_pred_gkpinn = u_pred_gkpinn.reshape(-1, 1)

Error_gkpinn = np.abs(u_star - u_pred_gkpinn).reshape(X.shape)
Error_pinn = np.abs(u_star - u_pred_pinn).reshape(X.shape)
vmin = Error_gkpinn.max()
vmax = Error_pinn.max()
mean_gkpinn = np.mean(Error_gkpinn)
mean_pinn = np.mean(Error_pinn)
mean_gkpinn, mean_pinn, vmin, vmax


# In[55]:


import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
from matplotlib.font_manager import FontProperties

l2_error_gkpinn = np.linalg.norm(u_star - u_pred_gkpinn, 2) / np.linalg.norm(u_star, 2)
print('l2_error: %e' % (l2_error_gkpinn))

Error_gkpinn = np.abs(u_star - u_pred_gkpinn).reshape(X.shape)

Z_gkpinn = u_pred_gkpinn.reshape(X.shape)

bounds = [mean_gkpinn, 0.0015, 0.0025, mean_pinn]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制3D曲面图
surf = ax.plot_surface(X, Y, Error_gkpinn, cmap='jet', edgecolor='none')
norm = mcolors.Normalize(vmin=Error_gkpinn.min(), vmax=Error_gkpinn.max())
cbar = fig.colorbar(surf, norm=norm, ticks=bounds)
cbar.ax.tick_params(labelsize=14)  # 设置刻度标签字体大小

# 设置坐标轴标签
ax.set_xlabel('x', labelpad=-2, fontsize=12)
ax.set_ylabel('y', labelpad=8, fontsize=12)

# 设置视角
ax.view_init(elev=30, azim=-70)

# 设置坐标轴刻度参数
ax.tick_params(axis='z', pad=4)
ax.tick_params(axis='x', pad=-5)
ax.tick_params(axis='y', pad=2)
ax.tick_params(which='major', labelsize=12)
ax.locator_params(axis='z', nbins=5)
plt.show()


# In[59]:


import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
from matplotlib.font_manager import FontProperties

l2_error_pinn = np.linalg.norm(u_star - u_pred_pinn, 2) / np.linalg.norm(u_star, 2)
print('l2_error: %e' % (l2_error_pinn))

Error_pinn = np.abs(u_star - u_pred_pinn).reshape(X.shape)

Z_pinn = u_pred_pinn.reshape(X.shape)

bounds = [mean_gkpinn, 0.0015, 0.0025, 0.5, 1.0, 1.5, vmax]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制3D曲面图
surf = ax.plot_surface(X, Y, Error_pinn, cmap='jet', edgecolor='none')
norm = mcolors.Normalize(vmin=Error_pinn.min(), vmax=Error_pinn.max())
cbar = fig.colorbar(surf, norm=norm, ticks=bounds)
cbar.ax.tick_params(labelsize=14)  # 设置刻度标签字体大小

# 设置坐标轴标签
ax.set_xlabel('x', labelpad=-2, fontsize=12)
ax.set_ylabel('y', labelpad=8, fontsize=12)

# 设置视角
ax.view_init(elev=30, azim=-70)

# 设置坐标轴刻度参数
ax.tick_params(axis='z', pad=4)
ax.tick_params(axis='x', pad=-5)
ax.tick_params(axis='y', pad=2)
ax.tick_params(which='major', labelsize=12)
ax.locator_params(axis='z', nbins=5)
plt.show()

