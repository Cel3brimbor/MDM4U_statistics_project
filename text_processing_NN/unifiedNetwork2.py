import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, TextVectorization, Embedding, GlobalAveragePooling1D #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.regularizers import l2 #type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

INITIAL_REGULARIZER = 0.0000
MAX_REGULARIZER = 5.0000
REGULARIZER_STEP = 0.1000
NEURONS = 512 

DATASET_NAME = "amazon_polarity_datasets_500k"
OUTPUT_JSON_FILE = f"{DATASET_NAME}_trainingResults/{NEURONS}_unified_network.json"

LEARNING_RATE = 0.001
BATCH_SIZE = 64
PATIENCE = 3
DROPOUT = 0.0

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 256
EMBEDDING_DIM = 64
NUM_CLASSES = 2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

physical_devices = tf.config.list_physical_devices('GPU')
print("\nPhysical devices:", physical_devices)
if physical_devices:
    print("\nGPU is detected. TensorFlow is ready to use metal acceleration\n")
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Failed to set memory growth: {e}")
else:
    print("\nGPU is not yet detected\n")

output_folder = DATASET_NAME
train_path = os.path.join(output_folder, DATASET_NAME+"_train.csv")
test_path = os.path.join(output_folder, DATASET_NAME + "_test.csv")

try:
    train_df = pd.read_csv(train_path, header=None, names=['label', 'text'])
    test_df = pd.read_csv(test_path, header=None, names=['label', 'text'])
except FileNotFoundError:
    print(f"Error: Required files ({train_path} or {test_path}) were not found in local directory.")
    exit()

train_df['text'] = train_df['text'].fillna('').astype(str)
test_df['text'] = test_df['text'].fillna('').astype(str)

all_texts = train_df['text'].tolist() + test_df['text'].tolist()
train_texts = train_df['text'].tolist()
train_labels = np.array(train_df['label'])
test_texts = test_df['text'].tolist()
test_labels = np.array(test_df['label'])

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.1, random_state=42
)

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

def build_model(regularizer_strength):
    """
    Constructs and returns a new Keras model with a specific L2 regularizer strength.
    """
    tf.keras.backend.clear_session() 

    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="text_input")

    x = Embedding(
        input_dim=VOCAB_SIZE, 
        output_dim=EMBEDDING_DIM, 
        input_length=MAX_SEQUENCE_LENGTH, 
        name="embedding_layer"
    )(inputs)

    x = GlobalAveragePooling1D(name="pooling_layer")(x)
    
    x = Dense(NEURONS, activation="relu", kernel_regularizer=l2(regularizer_strength), name="dense_1_l2")(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = Dense(NEURONS, activation="relu", kernel_regularizer=l2(regularizer_strength), name="dense_2_l2")(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = Dense(NEURONS, activation="relu", kernel_regularizer=l2(regularizer_strength), name="dense_3_l2")(x)
    
    outputs = Dense(1, activation="sigmoid", name="output_layer")(x)

    model = Model(inputs=inputs, outputs=outputs, name=f"L2_Classifier_R_{regularizer_strength}")
    return model


results = []
current_regularizer = INITIAL_REGULARIZER

while current_regularizer <= MAX_REGULARIZER:
    
    print("\n" + "="*70)
    print(f"--- Training with L2 Regularizer: {current_regularizer:.5f} ---")
    print("="*70)

    #build new
    model = build_model(current_regularizer)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True)
    ]

    history = model.fit(
        train_data,
        train_labels,
        batch_size=BATCH_SIZE,
        epochs=7, 
        validation_data=(val_data, val_labels),
        callbacks=callbacks,
        verbose=1 
    )
    
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=1)
    
    best_epoch = np.argmin(history.history['val_loss'])
    final_train_loss = history.history['loss'][best_epoch]
    final_train_accuracy = history.history['accuracy'][best_epoch]
    final_val_loss = history.history['val_loss'][best_epoch]
    final_val_accuracy = history.history['val_accuracy'][best_epoch]
    
    print(f"\nTraining stopped at epoch {len(history.history['loss'])}. Best epoch was {best_epoch + 1}.")
    print("\n--- Final Metrics (Best Epoch) ---")
    print(f"L2 Regularizer: {current_regularizer:.4f}")
    print(f"Training Loss: {final_train_loss:.4f} | Training Accuracy: {final_train_accuracy*100:.2f}%")
    print(f"Validation Loss: {final_val_loss:.4f} | Validation Accuracy: {final_val_accuracy*100:.2f}%")
    print(f"Evaluated Loss: {test_loss:.4f} | Evaluated Accuracy: {test_acc*100:.2f}%")

    result_entry = {
        "l2_regularizer": current_regularizer,
        "best_epoch": int(best_epoch + 1),
        "train_loss": f"{float(final_train_loss):.4f}",
        "train_accuracy": f"{float(final_train_accuracy) * 100:.2f}%",
        "eval_loss": f"{float(test_loss):.4f}",
        "eval_accuracy": f"{float(test_acc) * 100:.2f}%"
    }
    results.append(result_entry)

    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to '{OUTPUT_JSON_FILE}'")

    current_regularizer = round(current_regularizer + REGULARIZER_STEP, 5) 
    
    del model
    tf.keras.backend.clear_session()

print("\n" + "#"*70)
print("Experiment finished. All results are in:", OUTPUT_JSON_FILE)
print("#"*70)