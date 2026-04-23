import matplotlib.pyplot as plt

# Plot the loss for both training and validation
def plot_history(train_history):

    plt.title("Training Loss")
    plt.ylabel("Loss")
    plt.xlabel('Epoch')
    plt.plot(train_history['loss'])
    plt.show()

    plt.title("Validation Loss")
    plt.ylabel("Loss")
    plt.xlabel('Epoch')
    plt.plot(train_history['val_loss'])
    plt.show()

    plt.title("Accuracy Curve")
    plt.ylabel("Accuracy")
    plt.xlabel('Epochs')
    plt.plot(train_history['accuracy'],label='Training Accuracy')
    plt.show()

    plt.title("Accuracy Curve")
    plt.ylabel("Accuracy")
    plt.xlabel('Epochs')
    plt.plot(train_history['val_accuracy'],label='Validation Accuracy')
    plt.show()