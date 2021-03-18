import pandas
import matplotlib.pyplot as plt
import numpy as np

data = pandas.read_csv("insurance.csv", index_col=0)

print(data.axes)

x = list(data.axes[0])
y = list(data["charges"])
print(type(y))


# y = ax + b
LEARNING_RATE = 0.01
weights = [1, 1]
for i in range(len(x)):
    x1, y1 = x[i], y[i]
    pred = weights[0] * x1 + weights[1]
    if pred > y1:
        weights[0] -= pred * LEARNING_RATE
        weights[1] -= LEARNING_RATE
    elif pred < y1:
        weights[0] += pred * LEARNING_RATE
        weights[1] += LEARNING_RATE

plt.scatter(x, y)
plt.plot([x for x in range(100)], [x * weights[0] + weights[1] for x in range(100)])

plt.ylabel("Insurance charges")
plt.xlabel("Age")
plt.title(str(weights))
plt.show()
# https://github.com/stedy/Machine-Learning-with-R-datasets