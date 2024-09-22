import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-50, 50, 50)
y = np.linspace(-50, 50, 50)
z = np.linspace(-50, 50, 50)
X, Y, Z = np.meshgrid(x, y, z)

inequality_1 = (-0.2*X + 0.5*Y + Z < 0)
inequality_2 = (0.2*X - 0.7*Y + Z < 0)
inequality_3 = (0.8*X - 0.8*Y + Z >= 0)
inequality_4 = (0.8*X + Y + Z >= 0)

solution_space = inequality_1 & inequality_2 & inequality_3 & inequality_4

X_sol = X[solution_space]
Y_sol = Y[solution_space]
Z_sol = Z[solution_space]

ax.scatter(X_sol, Y_sol, Z_sol, color='blue', alpha=0.5)

ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_zlabel('b')

plt.show()
