"""
TANIM DOCTOR — train.py
This is the machine learning part. run this ONCE, it makes the model, then the app uses it forever.

what it does:
  takes the plantvillage leaf photos -> teaches a model to tell diseases apart -> saves it

we're using TRANSFER LEARNING: instead of teaching a brain from zero (needs a supercomputer
and weeks), we grab MobileNetV2 which already knows how to "see" shapes/textures from
millions of photos, and we just retrain the last layer to know plant diseases. way faster,
runs fine on a normal laptop.

BEFORE RUNNING:
  1. download the dataset from kaggle:
     https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
  2. unzip it so you have folders like:
     data/PlantVillage/Tomato___Early_blight/  (each folder = one disease, full of photos)
  3. point DATA_DIR below at that folder
  4. python train.py     <- then go eat, this takes a while

run:  python train.py
"""

import os, json
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------- settings you can tweak ----------
DATA_DIR   = "data/plantvillage dataset/color"   # where the disease folders live
IMG_SIZE   = 224                   # mobilenet wants 224x224
BATCH      = 32                    # how many photos it looks at per step. lower this if your pc runs out of memory
EPOCHS     = 8                     # how many passes over the data. more = better but slower
MODEL_OUT  = "model/tanim_model.keras"
LABELS_OUT = "model/labels.json"

os.makedirs("model", exist_ok=True)


def build_datasets():
    """split the photos into a training pile and a validation pile (80/20)"""
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(
            f"\n!! can't find '{DATA_DIR}'\n"
            "   download PlantVillage from kaggle and unzip it there first.\n"
            "   see the comment at the top of this file.\n")

    train = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="training", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH)

    val = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR, validation_split=0.2, subset="validation", seed=123,
        image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH)

    class_names = train.class_names
    print(f"\nfound {len(class_names)} disease classes")

    # save the label list so the app knows what index 0,1,2... means
    with open(LABELS_OUT, "w") as f:
        json.dump(class_names, f, indent=2)

    # speed things up — keep data flowing so the gpu/cpu never waits
    AUTO = tf.data.AUTOTUNE
    train = train.cache().shuffle(1000).prefetch(AUTO)
    val   = val.cache().prefetch(AUTO)
    return train, val, class_names


def build_model(n_classes):
    """mobilenet body (frozen) + our own little head on top"""
    # randomly flip/rotate/zoom the training photos so the model doesn't just
    # memorize them — it has to actually learn what a disease looks like
    augment = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.1),
    ], name="augment")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False          # freeze it — we only train our new head

    inputs  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)   # mobilenet wants pixels in -1..1
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)                                   # helps it not overfit
    outputs = layers.Dense(n_classes, activation="softmax")(x)    # one probability per disease

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model, base


def main():
    train, val, classes = build_datasets()
    model, base = build_model(len(classes))

    print("\n=== STAGE 1: training the head ===")
    model.fit(train, validation_data=val, epochs=EPOCHS,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

    # fine-tuning: unfreeze the top of mobilenet and nudge it gently with a tiny
    # learning rate. this is what pushes accuracy from "okay" to "actually good".
    print("\n=== STAGE 2: fine-tuning ===")
    base.trainable = True
    for layer in base.layers[:-30]:      # only the last 30 layers get to learn
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=4,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

    model.save(MODEL_OUT)
    loss, acc = model.evaluate(val)
    print(f"\n DONE. accuracy on unseen photos: {acc*100:.1f}%")
    print(f"   model saved -> {MODEL_OUT}")
    print(f"   labels saved -> {LABELS_OUT}")
    print("\n   now run:  python app.py")


if __name__ == "__main__":
    main()
