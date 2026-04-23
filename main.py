from data.download_dataset import download_data
from training.train_vgg import train_model
from evaluation.evaluate_vgg import evaluate_model
from utils.plot_utils import plot_history
from inference.predict_damage import test_model_on_image
from inference.caption_image import run_caption

def main():

    download_data()

    model, history = train_model()

    train_history = model.history.history

    plot_history(train_history)

    test_generator = evaluate_model(model)

    test_model_on_image(model, test_generator, index_to_plot=1)

    run_caption("aircraft_damage_dataset_v1/test/dent/149_22_JPG_jpg.rf.4899cbb6f4aad9588fa3811bb886c34d.jpg")

if __name__ == "__main__":
    main()