import numpy as np

a = [1,2,3,4,5,6,7,8]

a = np.array(a)
a = np.reshape(a, [-1,2])
a = a.transpose()
print(a)