import matplotlib.pyplot as plt

epochs = [1,2,3,4,5]
accuracy = [0.9, 0.92, 0.99, 0.98, 0.99]
val_accuracy = [0.7, 0.89, 0.97, 0.95, 0.98]

plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.title("OpenCV accuracy(yawn, no_yawn)")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0.5, 1)
plt.legend()
plt.show()