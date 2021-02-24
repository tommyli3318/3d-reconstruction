import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# X, Y, Z = self.xyz[:,:,0], self.xyz[:,:,1], self.xyz[:,:,2]

# print(X.shape)
# print(Y.shape)
# print(Z.shape)

# ax.scatter(X,Y,Z)

ax.scatter([1,1,3], [1,3,1], [3,1,1])
plt.show()