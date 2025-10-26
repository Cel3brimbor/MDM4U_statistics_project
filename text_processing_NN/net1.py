import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, TextVectorization, Embedding, GlobalAveragePooling1D #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.regularizers import l2 #type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

REGULARIZERS = 0.000002


DATASET_NAME = "imdb"
LEARNING_RATE = 0.001
BATCH_SIZE = 64
PATIENCE = 10
DROPOUT = 0.0

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 256
EMBEDDING_DIM = 64
NUM_CLASSES = 2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# check for GPU
physical_devices = tf.config.list_physical_devices('GPU')
print("\nPhysical devices:", physical_devices)
if physical_devices:
    print("\nGPU is detected. TensorFlow is ready to use metal acceleration\n")
    _ = input("Press enter to proceed: ")
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Failed to set memory growth: {e}")
else:
    print("\nGPU is not yet detected\n")


#load dataset
output_folder = DATASET_NAME+"_datasets"
train_path = os.path.join(output_folder, DATASET_NAME+"_train.csv")
test_path = os.path.join(output_folder, DATASET_NAME + "_test.csv")

try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
except FileNotFoundError:
    print("Files were not found in local directory")
    exit()

all_texts = train_df['text'].tolist() + test_df['text'].tolist()
train_texts = train_df['text'].tolist()
train_labels = np.array(train_df['label'])
test_texts = test_df['text'].tolist()
test_labels = np.array(test_df['label'])

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, random_state=42
)

#turn text into numerical values
vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH
)

vectorize_layer.adapt(all_texts)

print(f"Vocabulary size: {len(vectorize_layer.get_vocabulary())}")

train_data = vectorize_layer(np.array(train_texts)).numpy()
val_data = vectorize_layer(np.array(val_texts)).numpy()
test_data = vectorize_layer(np.array(test_texts)).numpy()


#build network
def build_model():
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="text_input")

    x = Embedding(
        input_dim=VOCAB_SIZE, 
        output_dim=EMBEDDING_DIM, 
        input_length=MAX_SEQUENCE_LENGTH, 
        name="embedding_layer"
    )(inputs)

    x = GlobalAveragePooling1D(name="pooling_layer")(x)
    
    x = Dense(256, activation="relu", kernel_regularizer=l2(REGULARIZERS), name="dense_1_l2")(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(REGULARIZERS), name="dense_2_l2")(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = Dense(64, activation="relu", kernel_regularizer=l2(REGULARIZERS), name="dense_3_l2")(x)
    outputs = Dense(1, activation="sigmoid", name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name="L2_Regularized_Classifier")
    return model

model = build_model()
model.summary()


print("\n--- Compiling and Training the Model ---")

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

#binary classification
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
]

history = model.fit(
    train_data,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=100,
    validation_data=(val_data, val_labels),
    callbacks=callbacks,
    verbose=1
)

#print training evaluation
final_train_loss = history.history['loss'][-1]
final_train_accuracy = history.history['accuracy'][-1]

print("\n--- Final Training Metrics ---")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Training Accuracy: {final_train_accuracy*100:.2f}%")

#print evaluation
print("\n--- Evaluation on New Data ---")
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=0)
print(f"\nEvalulated Loss: {test_loss:.4f}")
print(f"\nEvaluated Accuracy: {test_acc*100:.2f}%")
print(f"\nModel trained with L2 regularization strength (lambda coefficient): {REGULARIZERS}")