# scripts/constants.py

# This is the "Master List" of your classes.
# We use numbers (0, 1, 2) because the AI works with numbers.
LABELS = {
    0: "background",
    1: "crack",
    2: "mechanical_noise"
}

# This helper creates a reverse look-up (e.g., "crack" -> 1)
LABEL_MAP = {v: k for k, v in LABELS.items()}

# A quick way to get the number 3 (total classes)
NUM_CLASSES = len(LABELS)