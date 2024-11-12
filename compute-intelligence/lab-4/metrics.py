from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def evaluate_and_plot_confusion_matrix(model, X_test, Y_test, class_labels):
    """
    Evaluates the model on the test set and plots the confusion matrix.

    Parameters:
    model (tf.keras.Model): The trained model to evaluate.
    X_test (numpy.ndarray): Test images.
    Y_test (numpy.ndarray): True labels for the test images, one-hot encoded.
    class_labels (list): List of class labels for display in the confusion matrix.
    """
    # Evaluate model on test data
    test_loss, test_accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Get predictions for the test set
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true_classes = np.argmax(Y_test, axis=1)

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(Y_true_classes, Y_pred_classes)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
