"""
TANIM DOCTOR — app.py
the web app. takes a leaf photo, runs it through the model we trained, shows what's wrong
and what to do about it.

needs:  a trained model (run train.py first) + MySQL running (XAMPP)

run:  python app.py   ->  http://127.0.0.1:5000
"""

import os, json, uuid
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import numpy as np

import knowledge

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS = os.path.join(APP_DIR, "static", "uploads")
MODEL_PATH  = os.path.join(APP_DIR, "model", "tanim_model.keras")
LABELS_PATH = os.path.join(APP_DIR, "model", "labels.json")
ALLOWED = {"png", "jpg", "jpeg", "webp"}
IMG_SIZE = 224

# ---------- MySQL settings — change these to match your XAMPP setup ----------
DB = {
    "host": "localhost",
    "user": "root",
    "password": "",            # XAMPP's default root password is blank
    "database": "tanim_doctor",
}

app = Flask(__name__)
app.secret_key = "change-me-to-something-random"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16MB, phone photos are chunky
os.makedirs(UPLOADS, exist_ok=True)

# model gets loaded once on first use, not on every request (that'd be painfully slow)
_model = None
_labels = None


# ============ database ============
def get_db():
    import mysql.connector
    return mysql.connector.connect(**DB)

def init_db():
    """make the database and tables if they aren't there yet"""
    import mysql.connector
    cfg = {k: v for k, v in DB.items() if k != "database"}
    con = mysql.connector.connect(**cfg)
    cur = con.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB['database']} CHARACTER SET utf8mb4")
    cur.execute(f"USE {DB['database']}")
    cur.execute("""CREATE TABLE IF NOT EXISTS users(
        id INT AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(190) UNIQUE NOT NULL,
        name VARCHAR(120) NOT NULL,
        pw_hash VARCHAR(255) NOT NULL,
        created DATETIME)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS scans(
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NULL,
        image VARCHAR(190),
        plant VARCHAR(120),
        disease VARCHAR(160),
        confidence FLOAT,
        healthy TINYINT,
        created DATETIME,
        INDEX(user_id))""")
    con.commit(); cur.close(); con.close()


# ============ the model ============
def load_model():
    global _model, _labels
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "No trained model found. Run `python train.py` first — see README.")
        import tensorflow as tf
        _model = tf.keras.models.load_model(MODEL_PATH)
        with open(LABELS_PATH) as f:
            _labels = json.load(f)
    return _model, _labels


def predict(image_path):
    """run one photo through the model, give back the top 3 guesses"""
    import tensorflow as tf
    model, labels = load_model()

    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

    probs = model.predict(arr, verbose=0)[0]
    top3 = probs.argsort()[-3:][::-1]        # three highest-scoring classes
    return [{"label": labels[i], "conf": float(probs[i]) * 100} for i in top3]


# ============ helpers ============
def current_user():
    uid = session.get("uid")
    if not uid: return None
    con = get_db(); cur = con.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE id=%s", (uid,))
    u = cur.fetchone(); cur.close(); con.close()
    return u

@app.context_processor
def inject():
    return {"me": current_user()}


# ============ routes ============
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/diagnose", methods=["POST"])
def diagnose():
    file = request.files.get("image")
    if not file or not file.filename:
        flash("Pick a photo of the leaf first.", "warn")
        return redirect(url_for("index"))

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED:
        flash("Photo must be a JPG, PNG or WEBP.", "warn")
        return redirect(url_for("index"))

    fname = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOADS, secure_filename(fname))
    file.save(path)

    # shrink big phone photos so everything stays fast
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((1000, 1000))
        im.save(path, quality=88)
    except Exception:
        pass

    try:
        guesses = predict(path)
    except FileNotFoundError as e:
        flash(str(e), "warn")
        return redirect(url_for("index"))

    best = guesses[0]
    info = knowledge.lookup(best["label"])
    info["confidence"] = round(best["conf"], 1)
    info["image"] = fname
    info["others"] = [
        {**knowledge.lookup(g["label"]), "confidence": round(g["conf"], 1)} for g in guesses[1:]
    ]

    # save the scan so the user can look back at it
    try:
        con = get_db(); cur = con.cursor()
        cur.execute("""INSERT INTO scans(user_id,image,plant,disease,confidence,healthy,created)
                       VALUES(%s,%s,%s,%s,%s,%s,%s)""",
                    (session.get("uid"), fname, info["plant"], info["disease"],
                     info["confidence"], 1 if info["healthy"] else 0, datetime.now()))
        con.commit(); cur.close(); con.close()
    except Exception as e:
        print("couldn't save scan:", e)   # don't break the result page over this

    return render_template("result.html", r=info)


@app.route("/history")
def history():
    if not session.get("uid"):
        flash("Log in to see your scan history.", "warn")
        return redirect(url_for("login"))
    con = get_db(); cur = con.cursor(dictionary=True)
    cur.execute("SELECT * FROM scans WHERE user_id=%s ORDER BY created DESC LIMIT 100", (session["uid"],))
    rows = cur.fetchall(); cur.close(); con.close()
    return render_template("history.html", rows=rows)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        name  = (request.form.get("name") or "").strip()
        pw    = request.form.get("password") or ""
        if "@" not in email or not name or len(pw) < 6:
            flash("Fill everything in — password needs 6+ characters.", "warn")
            return render_template("register.html")
        con = get_db(); cur = con.cursor()
        try:
            cur.execute("INSERT INTO users(email,name,pw_hash,created) VALUES(%s,%s,%s,%s)",
                        (email, name, generate_password_hash(pw), datetime.now()))
            con.commit()
        except Exception:
            flash("That email is already registered.", "warn")
            return render_template("register.html")
        finally:
            cur.close(); con.close()
        flash("Account created — log in.", "ok")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        pw    = request.form.get("password") or ""
        con = get_db(); cur = con.cursor(dictionary=True)
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        u = cur.fetchone(); cur.close(); con.close()
        if u and check_password_hash(u["pw_hash"], pw):
            session["uid"] = u["id"]
            return redirect(url_for("index"))
        flash("Wrong email or password.", "warn")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.errorhandler(413)
def too_big(e):
    flash("That photo's too big (max 16MB). Try a smaller one.", "warn")
    return redirect(url_for("index"))


if __name__ == "__main__":
    try:
        init_db()
        print("database ready")
    except Exception as e:
        print("!! couldn't reach MySQL — is XAMPP running?\n  ", e)
    app.run(host="127.0.0.1", port=5000, debug=True)
