import keras.callbacks
import keras.losses
import keras.metrics
import keras.optimizers
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
from keras import layers
from keras.utils import plot_model

from src import logger
from src.data.generator import DataGenerator
from src.data.preprocess import get_data_sequence
from src.model.create_model import make_model
from src.utils import check_path


def train_or_evaluate(
        work_mode: str,
        model_input_data_shape,
        output_save_path: str,
        tensorboard_log_dir: str = "tf-logs",
        checkpoint_index = None,
        batch_size=16,
        epochs=10,
        MCD_type="047",
        activation=layers.ReLU(),
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()],
):
    # template filename
    checkpoint_filename = "checkpoint_at_{epoch}.keras"
    # check all paths that will be used
    checkpoints_path = f"{output_save_path}/checkpoints"
    output_data_save_path = f"{output_save_path}/prediction"
    model_structure_pic_save_path = f"{output_save_path}/model_structure"
    check_path(checkpoints_path, model_structure_pic_save_path, output_data_save_path)

    # create model
    model = make_model(model_input_data_shape, activation=activation)
    # # load weight if in evaluation mode
    if work_mode == "evaluate":
        checkpoint_file_path = checkpoints_path + "/" + checkpoint_filename.format(epoch=checkpoint_index)
        model.load_weights(checkpoint_file_path)
    # compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # plot model structure
    # plot_model(model, to_file=f"{model_structure_pic_save_path}/model.png", show_shapes=True, show_layer_names=True)
    logger.debug(f"Save model structure picture to {model_structure_pic_save_path}/model.png")

    years = [2021]
    time_step = 1
    train_data, val_data, test_data = get_data_sequence(years, time_step, MCD_type)

    # train or evaluate
    if work_mode == "train":
        # create data generator
        train_data_generator = DataGenerator(train_data, batch_size, shuffle=True, name="Train")
        val_data_generator = DataGenerator(val_data, batch_size, shuffle=True, name="Val")
        # callbacks to save checkpoints and tensorboard logs
        callbacks = [
            keras.callbacks.ModelCheckpoint(f"{checkpoints_path}/{checkpoint_filename}", verbose=1),
            keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
        ]
        # train
        model.fit_generator(generator=train_data_generator, validation_data=val_data_generator, epochs=epochs,
                            use_multiprocessing=True, workers=6, callbacks=callbacks)

        return None
    elif work_mode == "evaluate":
        # create data generator
        evaluate_generator = DataGenerator(test_data, batch_size, shuffle=False, name="Evaluate")

        index = 0
        
        check_path("pics")
        
        # test
        for i in range(len(evaluate_generator)):
            X, Y = evaluate_generator[i]
            _labels_name = evaluate_generator.labels_name
            
            predicts = model.predict(X)
            # index = 0
            for _predicts, labels, name in zip(predicts, Y, _labels_name):                
                # save prediction
                np.savez(f"{output_data_save_path}/{name}.npz", predict=_predicts, label=labels)

                # plot
                fig = plt.figure(figsize=(16, 10))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                ax1.imshow(labels, vmin=0, vmax=40)
                im = ax2.imshow(_predicts, vmin=0, vmax=40)
                ax1.set_title("(a) labels", loc="left")
                ax2.set_title("(b) predicts", loc="left")
                ax_position = ax2.get_position()
                _, y0, x1, y1 = ax_position.x0, ax_position.y0, ax_position.x1, ax_position.y1
                cax1_position = Bbox.from_extents(x1 + 0.02, y0, x1 + 0.04, y1)
                cax = fig.add_axes(cax1_position)
                fig.colorbar(mappable=im, cax=cax)
                # save picture
                fig.savefig(f"pics/{name}.png")
                plt.clf()
                # exit(0)

                index += 1

        return None


__all__ = ['train_or_evaluate']
