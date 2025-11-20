import keras.optimizers
import keras.losses
import keras.metrics

from keras import layers

from src.train.run import train_or_evaluate

if __name__ == "__main__":
    
    train_or_evaluate(
        work_mode="evaluate",
        model_input_data_shape=((256, 256, 1), ),
        output_save_path="output",
        tensorboard_log_dir="tf-logs",
        checkpoint_index=5,
        epochs=10,
        batch_size=8,
        MCD_type="047",
        activation=layers.ReLU(),
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
