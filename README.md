# 🌿 Tanim Doctor

Photograph a sick plant leaf, get told what disease it likely has and what to do about it. Runs a **computer-vision model you train yourself** — no API key, no monthly cost, works offline once trained.

Built with Flask + MySQL + TensorFlow.

---

## Setup — do these in order

### 1. Install the libraries

**Windows (PowerShell):**
```powershell
py -m venv venv
venv\Scripts\pip install -r requirements.txt
```

**Ubuntu / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

TensorFlow is a big download (a few hundred MB) — be patient on the first install.

### 2. Get the dataset

Download PlantVillage from Kaggle (free account needed):
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

Unzip it into the `data/` folder so it looks like this:

```
data/PlantVillage/
    Apple___Apple_scab/          <- each folder is one disease
    Apple___Black_rot/
    Tomato___Late_blight/
    Tomato___healthy/
    ... (38 folders total, ~54,000 photos)
```

If your unzipped folder is named differently, just edit `DATA_DIR` at the top of `train.py`.

### 3. Train the model (do this once)

```powershell
venv\Scripts\python train.py
```

**This takes a while** — roughly 1–3 hours on a normal laptop CPU, much faster if you have a good GPU. It's doing real machine learning. Go eat, watch something, come back.

When it finishes you'll see the accuracy and two new files in `model/`. Typical accuracy is 95%+.

Low on time or patience? Lower `EPOCHS` in `train.py` from 8 to 3 — less accurate, much faster.

### 4. Start MySQL

Open XAMPP and start **MySQL**. The app creates the database and tables itself on first run. If your MySQL has a root password, put it in the `DB` settings near the top of `app.py`.

### 5. Run it

```powershell
venv\Scripts\python app.py
```

Open **http://127.0.0.1:5000**, upload a leaf photo, get a diagnosis.

---

## How it works

**`train.py`** — uses *transfer learning*: takes MobileNetV2 (a model that already knows how to see shapes and textures from millions of photos) and retrains just the final layer to recognise plant diseases. That's why it trains in hours instead of weeks. Stage 2 then fine-tunes the last 30 layers to push accuracy higher.

**`knowledge.py`** — the model only outputs a label like `Tomato___Early_blight`. This file turns that into real advice: what the disease is, treatment steps, prevention.

**`app.py`** — the web app. Loads the trained model once, resizes uploaded photos, runs the prediction, saves every scan to MySQL.

---

## Honest limits

- It only knows the ~38 disease classes in PlantVillage. Show it something outside that and it'll still guess — that's why the app always shows a **confidence %** and alternative possibilities.
- Photo quality matters enormously. One leaf, good daylight, plain background.
- The app tells users to confirm with an agriculture office before spending money on treatment. Keep that in — bad advice can cost a farmer a crop.

## Ideas to extend it

- Add more languages to the advice in `knowledge.py`
- A "my plants" tracker to watch a plant over time
- Let users flag wrong diagnoses, so you can improve the model
- Offline mobile version using TensorFlow Lite
