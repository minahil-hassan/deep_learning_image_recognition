import os
import json
import nbformat
from nbformat.v4 import new_notebook, new_code_cell

# Prepare config list
config_dir = "experiment_configs"
config_files = sorted([f for f in os.listdir(config_dir) if f.endswith(".json")])

# ---------- Top cells: shared setup ----------
top_cells = [
    new_code_cell("""# Imports
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import DenseNet121
import tensorflow_datasets as tfds"""),

    new_code_cell("""# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)"""),

    new_code_cell("""# Data Preprocessing
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds, val_ds = tfds.load('caltech101', split=['train[:80%]', 'train[80%:]'], as_supervised=True)
train_ds = train_ds.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

num_classes = tfds.builder('caltech101').info.features['label'].num_classes"""),

    new_code_cell("""# Model Builder
def build_densenet_model(cfg):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = cfg["train_entire_model"]
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(cfg["dense_units"], activation=cfg["activation"])(x)
    x = layers.Dropout(cfg["dropout_rate"])(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=output)

    if cfg["optimiser"] == 'adam':
        opt = optimizers.Adam(learning_rate=cfg["learning_rate"])
    elif cfg["optimiser"] == 'adagrad':
        opt = optimizers.Adagrad(learning_rate=cfg["learning_rate"])
    elif cfg["optimiser"] == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=cfg["learning_rate"], momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimiser: {cfg['optimiser']}")

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model"""),

    new_code_cell("""# Training Plot
def plot_training(history, name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"{name} Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{name} Loss")
    plt.legend()

    plt.savefig(f"plots/{name}.png")
    plt.close()""")
]

# ---------- Experiment cells ----------
experiment_cells = []
for cfg_file in config_files:
    exp_name = os.path.splitext(cfg_file)[0]
    cell_code = f"""
# Experiment: {exp_name}

# Load config
with open("experiment_configs/{exp_name}.json", "r") as f:
    cfg = json.load(f)

# Skip if model already exists
model_path = f"models/{{cfg['name']}}.keras"
if os.path.exists(model_path):
    print(f"Skipping {{cfg['name']}} — model already exists.")
    sys.exit()

# Build model and setup callbacks
model = build_densenet_model(cfg)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=cfg["num_epochs"],
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Plot training
plot_training(history, cfg['name'])

# Save result
final_result = {{
    "experiment": cfg["name"],
    "val_accuracy": history.history["val_accuracy"][-1],
    "val_loss": history.history["val_loss"][-1],
    "epochs_ran": len(history.history["val_loss"]),
}}
df = pd.DataFrame([final_result])
csv_path = "results/experiment_results.csv"
if os.path.exists(csv_path):
    df_existing = pd.read_csv(csv_path)
    df_combined = pd.concat([df_existing, df])
else:
    df_combined = df
df_combined.to_csv(csv_path, index=False)
print(f"Completed: {{cfg['name']}}")
"""
    experiment_cells.append(new_code_cell(cell_code.strip()))

# Combine top and experiment cells
nb = new_notebook(cells=top_cells + experiment_cells)

# Save the notebook
with open("experiment_runner.ipynb", "w") as f:
    nbformat.write(nb, f)

print("Modular experiment_runner.ipynb generated successfully.")
