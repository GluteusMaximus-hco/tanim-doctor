# Tanim Doctor

Take a photo of a sick plant leaf, find out what disease it has and what to do about it.

![Python](https://img.shields.io/badge/python-3.12-blue) ![TensorFlow](https://img.shields.io/badge/tensorflow-2.16-orange) ![Flask](https://img.shields.io/badge/flask-3.0-lightgrey) ![accuracy](https://img.shields.io/badge/val__accuracy-96.4%25-brightgreen)

---

## Why I built this

A lot of plant disease apps either want a subscription or fire your photo off to somebody's paid API. I wanted to know whether a student with a laptop and a free dataset could build something that actually works, and that keeps working with the wifi off.

Turns out yes. The model here is trained from scratch on my own machine. No API key, no per-request cost, nothing leaves the computer.

The other reason: the advice matters as much as the diagnosis. Telling a farmer "late blight" and nothing else is useless. Every result comes with what the disease actually is, what to do this week, and how to stop it coming back.

## What it does

- Upload a leaf photo, get a diagnosis with a confidence score
- Shows the top 3 possibilities, not just one, because the model is not always right and pretending otherwise is dangerous
- Treatment steps and prevention advice for each disease
- Optional account to keep a history of your scans

Covers 38 classes across 14 plants: tomato, potato, corn, apple, grape, peach, pepper, cherry, strawberry, orange, squash, soybean, raspberry, blueberry.

## How it works

Three pieces.

**`train.py`** does the machine learning. It uses transfer learning on MobileNetV2 rather than training a network from zero, because training from zero on a laptop CPU would take weeks and produce something worse. MobileNetV2 already knows how to see edges, textures, colour patterns from ImageNet, so I freeze all of that and retrain only the final classification layer on leaves. Then stage two unfreezes the last 30 layers and nudges them with a very low learning rate (1e-5), which is what takes it from roughly 94% to 96.4%.

**`knowledge.py`** is the part people forget. The model outputs a string like `Tomato___Early_blight`. That means nothing to a person holding a dying plant. This file maps every class to a real explanation, treatment steps, and prevention advice.

**`app.py`** is the Flask web app. Loads the model once at startup instead of per-request, resizes uploads so a 12MB phone photo doesn't slow everything down, and logs each scan to MySQL.

## Results

Validation accuracy: **96.4%** on images the model never saw during training.

Training ran about 50 minutes on CPU. EarlyStopping ended stage one at epoch 7 because validation accuracy had stopped improving.

Worth being clear about what that number means. 96.4% is on the PlantVillage dataset, where every leaf is photographed against a clean background in good lighting. A real photo taken in a garden with dirt, shadows, and three overlapping leaves is a harder problem, and accuracy in the field is lower. See limitations.

## Setup

You need Python 3.12. Not 3.13 or 3.14, TensorFlow does not support them yet and you will hit a wall on install.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Get the dataset from [PlantVillage on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and unzip it into `data/`. It ships with `color`, `grayscale`, and `segmented` versions. Use `color`, it is the one that resembles a real photo. Point `DATA_DIR` in `train.py` at it.

Train the model. This takes roughly an hour on CPU:

```bash
python train.py
```

Start MySQL, then run the app. It creates its own database and tables on first run:

```bash
python app.py
```

Open http://localhost:5000.

If you are low on patience, drop `EPOCHS` from 8 to 3. You lose a few percent of accuracy and save most of the time.

## Honest limitations

**It will confidently misdiagnose plants it has never seen.** The model knows 38 classes. Show it a mango leaf and it will still return an answer, because softmax always sums to 1 and something has to win. It does not know what it does not know. This is why the app always shows a confidence number and alternatives instead of a single verdict.

**Dataset photos are cleaner than real ones.** Every PlantVillage image is a single leaf on a plain background. Field photos have soil, shadows, multiple leaves, weird angles. Expect lower confidence on real photos than the 96.4% suggests.

**Two diseases can look nearly identical.** Early blight and target spot both produce brown rings. Even agronomists sometimes need a lab. The model guesses from pixels alone and sometimes guesses wrong.

**Wrong advice costs money.** If someone sprays the wrong fungicide on the strength of this app, that is wasted money and a crop that keeps dying. The disclaimer telling users to confirm with their local agriculture office is deliberate and should stay.

**No web deployment yet.** Runs locally. Making it public would mean hosting, and TensorFlow needs more memory than most free tiers allow.

## Things I want to add

- **Rejection threshold.** If the top prediction is below roughly 60%, say "I am not sure, try a clearer photo" instead of guessing. This is the fix that matters most.
- **Filipino translations** in `knowledge.py`. The structure already supports it, the text just needs writing.
- **Track a plant over time.** Photograph the same plant weekly and see whether treatment is working. This is where accounts would finally earn their place, since right now they only store history.
- **Report wrong diagnoses.** Let users flag bad results, collect them, retrain on real field photos instead of clean dataset ones.
- **TensorFlow Lite** so it runs on a phone with no server at all. That is the version a farmer in a field would actually use.
- **More plants.** Rice and banana are not in PlantVillage and matter a lot here.

## Built with

Python, TensorFlow/Keras, Flask, MySQL, Pillow. Trained on the PlantVillage dataset.
