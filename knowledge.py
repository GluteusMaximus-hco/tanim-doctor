"""
knowledge.py — what to actually DO about each disease.

the model only spits out a label like "Tomato___Early_blight". that's useless to a farmer
on its own. this file turns that label into real advice: what it is, what to do now,
how to stop it coming back.

advice here is general good practice. for a serious outbreak people should still talk to
their local agri office — we say that in the app too.
"""

# fallback used when the model predicts a class we haven't written advice for yet
GENERIC = {
    "about": "A plant disease was detected on this leaf.",
    "treat": [
        "Remove and destroy the affected leaves — don't compost them.",
        "Avoid watering the leaves; water the soil at the base instead.",
        "Improve airflow by spacing or pruning the plants.",
        "Ask your local agriculture office about a suitable fungicide.",
    ],
    "prevent": [
        "Rotate crops each season instead of replanting the same thing.",
        "Use disease-free seeds or seedlings.",
        "Clear old plant debris from the field after harvest.",
    ],
}

HEALTHY = {
    "about": "This leaf looks healthy — no disease detected.",
    "treat": ["Nothing to treat. Keep doing what you're doing."],
    "prevent": [
        "Keep watering at the base, not on the leaves.",
        "Check your plants weekly so you catch problems early.",
        "Don't crowd plants — airflow prevents most leaf disease.",
    ],
}

# keyed by the disease part of the plantvillage folder name (lowercased)
DISEASES = {
    "early_blight": {
        "about": "Early blight is a fungus. Look for dark brown spots with rings inside them, like a target, usually starting on the older lower leaves.",
        "treat": [
            "Pick off and burn or bury the spotted leaves right away.",
            "Don't let water splash soil onto the leaves — mulch the base.",
            "Apply a copper-based or chlorothalonil fungicide, repeating as the label says.",
            "Keep the plant fed; a strong plant fights it off better.",
        ],
        "prevent": [
            "Rotate — don't plant tomatoes or potatoes in the same spot two seasons running.",
            "Space plants further apart so leaves dry quickly.",
            "Water in the morning at the base of the plant.",
        ],
    },
    "late_blight": {
        "about": "Late blight is serious and spreads fast, especially in cool wet weather. Look for large greasy-looking dark patches, sometimes with white fuzz underneath.",
        "treat": [
            "Act fast — this can wipe out a whole crop in days.",
            "Pull out badly infected plants completely and destroy them away from the field.",
            "Spray remaining plants with a fungicide labelled for late blight.",
            "Stop overhead watering entirely until it's under control.",
        ],
        "prevent": [
            "Plant resistant varieties if you can get them.",
            "Don't plant too densely — wet crowded leaves are how it spreads.",
            "Destroy leftover tubers and volunteer plants; they carry it over.",
        ],
    },
    "bacterial_spot": {
        "about": "A bacterial infection showing as small dark water-soaked spots that may have a yellow halo. It spreads through splashing water and handling wet plants.",
        "treat": [
            "Remove infected leaves and fruit; wash hands and tools after.",
            "Switch to drip or base watering — splashing spreads it.",
            "Copper-based sprays can slow it, but won't cure it.",
            "Don't work among the plants while they're wet.",
        ],
        "prevent": [
            "Use certified disease-free seed.",
            "Rotate crops for 2-3 years.",
            "Disinfect tools and stakes between seasons.",
        ],
    },
    "leaf_mold": {
        "about": "A fungus that loves humid, poorly-ventilated spaces. Pale yellow patches appear on top of the leaf with olive-green fuzzy mold underneath.",
        "treat": [
            "Increase airflow immediately — prune lower leaves, open the greenhouse.",
            "Remove affected leaves.",
            "Reduce humidity; avoid watering late in the day.",
            "Apply a suitable fungicide if it keeps spreading.",
        ],
        "prevent": ["Keep humidity down.", "Space plants generously.", "Grow resistant varieties where possible."],
    },
    "septoria_leaf_spot": {
        "about": "Many small round spots with dark edges and pale grey centers, usually starting low on the plant and moving up.",
        "treat": [
            "Strip off the infected lower leaves.",
            "Mulch the soil to stop spores splashing up.",
            "Apply fungicide on a regular schedule during wet weather.",
        ],
        "prevent": ["Clean up all plant debris after harvest.", "Rotate crops.", "Water at the base only."],
    },
    "spider_mites": {
        "about": "Not a disease — tiny pests. Leaves look stippled, dusty or bronzed, and you may see fine webbing underneath.",
        "treat": [
            "Spray the undersides of leaves hard with water to knock them off.",
            "Apply insecticidal soap or neem oil, covering the leaf undersides.",
            "Repeat every few days — they breed fast in hot dry weather.",
        ],
        "prevent": ["Keep plants well-watered; mites thrive on stressed dry plants.", "Check leaf undersides weekly."],
    },
    "target_spot": {
        "about": "Brown spots with concentric rings, similar to early blight but often on younger leaves and fruit too.",
        "treat": ["Remove affected leaves.", "Improve airflow.", "Apply an appropriate fungicide."],
        "prevent": ["Rotate crops.", "Avoid overhead irrigation.", "Clear debris after harvest."],
    },
    "yellow_leaf_curl_virus": {
        "about": "A virus spread by whiteflies. Leaves curl upward, turn yellow at the edges, and the plant stays stunted. There is no cure once infected.",
        "treat": [
            "There's no cure — remove and destroy infected plants so they don't infect others.",
            "Control the whiteflies: yellow sticky traps, insecticidal soap, neem.",
            "Check nearby plants daily.",
        ],
        "prevent": [
            "Use whitefly netting on seedlings.",
            "Plant resistant varieties.",
            "Keep the area free of weeds that host whiteflies.",
        ],
    },
    "mosaic_virus": {
        "about": "A virus giving leaves a mottled light-and-dark green patchy look, often with distorted growth. Spreads by touch and tools.",
        "treat": [
            "No cure — pull out and destroy infected plants.",
            "Wash hands and disinfect tools before touching healthy plants.",
            "Don't smoke near tomato plants; tobacco can carry related viruses.",
        ],
        "prevent": ["Use certified seed.", "Disinfect tools regularly.", "Control aphids, which spread it."],
    },
    "common_rust": {
        "about": "Reddish-brown powdery pustules on both sides of the leaf that rub off on your fingers.",
        "treat": ["Remove badly infected leaves.", "Apply a fungicide if it's spreading fast.", "Avoid overhead watering."],
        "prevent": ["Plant resistant varieties.", "Don't crowd plants.", "Rotate crops."],
    },
    "northern_leaf_blight": {
        "about": "Long grey-green cigar-shaped lesions on corn leaves that later turn tan.",
        "treat": ["Apply fungicide early if the crop is valuable.", "Remove crop debris after harvest."],
        "prevent": ["Rotate away from corn for a season.", "Plant resistant hybrids.", "Till in old residue."],
    },
    "cercospora_leaf_spot": {
        "about": "Small grey spots with reddish-brown borders, which can merge and kill large areas of leaf.",
        "treat": ["Remove affected leaves.", "Apply fungicide during humid weather.", "Improve airflow."],
        "prevent": ["Rotate crops.", "Avoid overhead watering.", "Clean up debris."],
    },
    "black_rot": {
        "about": "Brown circular lesions on leaves and shrivelled black fruit. A fungus that overwinters in old fruit left on the plant.",
        "treat": [
            "Remove all mummified fruit and infected canes — this is the key step.",
            "Prune for airflow.",
            "Apply fungicide from early leaf stage through fruiting.",
        ],
        "prevent": ["Clean up fallen fruit every season.", "Prune annually.", "Don't let fruit touch wet soil."],
    },
    "esca": {
        "about": "Also called black measles. Leaves show tiger-stripe yellow or red patterns between the veins; wood inside the vine decays.",
        "treat": [
            "No reliable cure — remove and destroy severely affected vines.",
            "Prune out dead wood, and seal large pruning wounds.",
            "Prune in dry weather to reduce infection.",
        ],
        "prevent": ["Avoid large pruning cuts.", "Disinfect pruning tools.", "Keep vines unstressed and well-watered."],
    },
    "leaf_blight": {
        "about": "Irregular brown dead patches on the leaf, often starting at the edges and spreading inward.",
        "treat": ["Remove affected leaves.", "Improve airflow and drainage.", "Apply a suitable fungicide."],
        "prevent": ["Rotate crops.", "Avoid wetting foliage.", "Remove debris after harvest."],
    },
    "haunglongbing": {
        "about": "Citrus greening — a very serious bacterial disease spread by a small insect (psyllid). Leaves yellow unevenly, fruit stays small, green and bitter.",
        "treat": [
            "There is no cure. Infected trees should be removed to protect nearby trees.",
            "Control psyllids aggressively.",
            "Report it — in many places this is a regulated disease.",
        ],
        "prevent": ["Buy certified disease-free trees.", "Monitor for psyllids constantly.", "Remove infected trees promptly."],
    },
    "powdery_mildew": {
        "about": "White powdery dust on the leaf surface, like someone sprinkled flour on it. Thrives in warm days with cool humid nights.",
        "treat": [
            "Remove the worst leaves.",
            "Spray with a potassium bicarbonate, sulfur, or neem-based solution.",
            "Increase sunlight and airflow.",
        ],
        "prevent": ["Space plants well.", "Don't over-fertilize with nitrogen.", "Choose resistant varieties."],
    },
    "scab": {
        "about": "Olive-green to brown velvety spots on leaves and corky scabs on fruit. Very common in wet springs.",
        "treat": ["Remove fallen leaves — that's where it overwinters.", "Apply fungicide from bud break through wet weather.", "Prune for airflow."],
        "prevent": ["Rake and destroy fallen leaves every autumn.", "Plant resistant varieties.", "Prune to open the canopy."],
    },
    "leaf_scorch": {
        "about": "Leaf edges turn brown and dry as if burnt, often with a purple or reddish border on the dead area.",
        "treat": ["Remove affected leaves.", "Make sure the plant gets consistent water.", "Apply fungicide if it spreads."],
        "prevent": ["Water evenly — don't let plants dry out then flood them.", "Mulch to hold moisture.", "Clear old leaves."],
    },
}


def _norm(raw_label):
    """
    plantvillage folders look like 'Tomato___Late_blight' or 'Apple___healthy'.
    split that into a readable plant name and a disease key we can look up.
    """
    parts = raw_label.split("___")
    plant = parts[0].replace("_", " ").strip()
    disease_raw = parts[1] if len(parts) > 1 else "unknown"
    disease_key = disease_raw.lower().replace(" ", "_").replace("-", "_")
    # strip junk some versions of the dataset include
    for junk in ["(including_sour)", "_(maize)", "_bell", "tomato_", "corn_", "grape_", "potato_", "pepper,_"]:
        disease_key = disease_key.replace(junk, "")
    disease_key = disease_key.strip("_")
    pretty = disease_raw.replace("_", " ").replace("  ", " ").strip().title()
    return plant, disease_key, pretty


def lookup(raw_label):
    """turn a model label into a full advice card"""
    plant, key, pretty = _norm(raw_label)

    if "healthy" in key:
        info = HEALTHY
        pretty = "Healthy"
    else:
        info = None
        # try exact, then loose match so small naming differences still hit
        if key in DISEASES:
            info = DISEASES[key]
        else:
            for k, v in DISEASES.items():
                if k in key or key in k:
                    info = v
                    break
        if info is None:
            info = GENERIC

    return {
        "plant": plant,
        "disease": pretty,
        "healthy": "healthy" in key,
        "about": info["about"],
        "treat": info["treat"],
        "prevent": info["prevent"],
    }
