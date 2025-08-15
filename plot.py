import matplotlib.pyplot as plt

# Example data
x = [1.0, 2.0, 3.0, 4.0]
# vanillaLasso = [0.19, 0.595, 0.82, 0.96]
# elasticNet = [0.21, 0.59, 0.81, 0.95]
# NNLS = [0.144, 0.55, 0.8583333333333333, 0.9416666666666667]
fusedLasso = [0.06451612903225806, 0.2696629213483146, 0.5121951219512195, 0.8902439024390244]

# plt.plot(x, vanillaLasso, marker='o', linestyle='-', color='b', label='Vanilla Lasso')
# plt.plot(x, elasticNet, marker='o', linestyle='-', color='r', label='Elastic Net')
# plt.plot(x, NNLS, marker='o', linestyle='-', color='g', label='NNLS')
plt.plot(x, fusedLasso, marker='o', linestyle='-', color='y', label='Fused Lasso')
plt.xlabel('# Delta')
plt.ylabel('TPR_cp')
plt.xticks([1.0, 2.0, 3.0, 4.0])
plt.ylim(0, 1.0)
plt.legend()
plt.show()
plt.savefig('./TPR.pdf')