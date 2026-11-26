import matplotlib.pyplot as plt
import random
import numpy as np
from classification_assignment import x_train, y_train, x_test, y_test, model, data_labels

def visualize(imgs, labels, preds=None):
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flatten()):
        ran_idx = random.randint(0, len(imgs) - 1)
        label_idx = labels[ran_idx][0]
        ax.imshow(imgs[ran_idx])

        if preds is None:
            ax.set_title(f"Label: {data_labels[label_idx]}")
        else:
            pred_idx = int(preds[ran_idx])
            ax.set_title(f"Pred: {data_labels[pred_idx]} | Real: {data_labels[label_idx]}")

        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize(x_train, y_train)

    pred_classes = np.argmax(model.predict(x_test), axis=1)
    visualize(x_test, y_test, preds=pred_classes)

