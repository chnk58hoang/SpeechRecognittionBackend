from django.apps import AppConfig
import tensorflow as tf
import tensorflow_text as tftext


def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    """Model similar to DeepSpeech2."""
    # Model's input
    input_spectrogram = tf.keras.layers.Input((None, input_dim), batch_size=None, name="input")
    # Expand the dimension to use 2D CNN.
    x = tf.keras.layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    # Convolution layer 1
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="conv_1_bn")(x)
    x = tf.keras.layers.ReLU(name="conv_1_relu")(x)
    # Convolution layer 2
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="conv_2_bn")(x)
    x = tf.keras.layers.ReLU(name="conv_2_relu")(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = tf.keras.layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = tf.keras.layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = tf.keras.layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = tf.keras.layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = tf.keras.layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = tf.keras.layers.ReLU(name="dense_1_relu")(x)
    x = tf.keras.layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = tf.keras.layers.Dense(units=output_dim + 1, activation="softmax")(x)
    # Model
    model = tf.keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model


def create_tokenizer(model_path):
    with open(model_path, 'rb') as f:
        model = f.read()

    tokenizer = tftext.SentencepieceTokenizer(
        model=model, out_type=tf.int32, nbest_size=0,
        add_bos=False, add_eos=False, return_nbest=False
    )

    return tokenizer


class SpeechConfig(AppConfig):
    default_auto_field = 'django.db.model_checkpoint.BigAutoField'
    name = 'Speech'

    "load our model only once here"
    model = build_model(
        input_dim=80,
        output_dim=54,
        rnn_units=128,
        rnn_layers=2,
    )

    model.load_weights('Speech/model_checkpoint/ckpt-epoch-66.ckpt')
    tokenizer = create_tokenizer('Speech/vocab/subword_54.model')
