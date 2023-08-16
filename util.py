import numpy as np
import matplotlib.pyplot as plt


def sigmoid(a):
    return 1/(1+np.exp(-a))


#it can be used for other models, not only for logistic regression model
def plot_decision_boundary_logistic_regression(model, tr_x, tr_y, val_x, val_y, acc):
    # Create a meshgrid of data points to calculate the decision boundary
    h = .02
    x_min, x_max = tr_x[:, 0].min() - 1, tr_x[:, 0].max() + 1
    y_min, y_max = tr_x[:, 1].min() - 1, tr_x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Calculate the decision boundary
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # Plot the training data points
    plt.scatter(tr_x[:, 0], tr_x[:, 1], c=tr_y, cmap=plt.cm.Spectral, marker='o', s=80, label='Training data')
    # Plot the validation data points
    plt.scatter(val_x[:, 0], val_x[:, 1], c=val_y, cmap=plt.cm.Spectral, facecolors='none', marker='x', s=80, label='Validation data')
    plt.text(7, 3.5, 'y=1', color='white', fontsize=12, fontweight='bold')
    plt.text(6.5, 4.5, 'y=0', color='white', fontsize=12, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title("acc: %f" %acc)
    plt.show()
