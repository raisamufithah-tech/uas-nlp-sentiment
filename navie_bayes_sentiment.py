import math

# =========================
# DATASET (diambil & disederhanakan dari dataset_sentimen.pdf)
# label: 0 = negatif, 1 = netral, 2 = positif
# =========================
data = [
    # POSITIF (2)
    ("materinya goks abis auto paham instruktornya asik parah", 2),
    ("platformnya interaktif banget jadi gak mager belajar", 2),
    ("mentornya pro player ngajarnya santuy tapi ngena", 2),
    ("ilmunya kepake banget di dunia kerja gak nyesel", 2),
    ("sertifikatnya nambah value buat cv", 2),

    # NEGATIF (0)
    ("materinya basi gak relate sama jaman sekarang", 0),
    ("platformnya sering error bikin emosi", 0),
    ("instrukturnya kurang interaktif bikin boring", 0),
    ("sertifikatnya gak diakui rugi banget", 0),
    ("harga mahal tapi kualitas zonk", 0),

    # NETRAL (1)
    ("materinya standar aja gak jelek tapi gak wow", 1),
    ("platformnya oke lah jarang error", 1),
    ("instrukturnya lumayan tapi kurang detail", 1),
    ("cukup oke buat nambah ilmu dasar", 1)
]

# =========================
# PREPROCESSING
# =========================
def preprocess(text):
    text = text.lower()
    text = text.replace(".", "")
    text = text.replace(",", "")
    return text.split()

# =========================
# PERSIAPAN DATA
# =========================
classes = set(label for _, label in data)

texts = []
labels = []

for text, label in data:
    texts.append(preprocess(text))
    labels.append(label)

# =========================
# HITUNG PRIOR
# =========================
prior = {}
for c in classes:
    prior[c] = labels.count(c) / len(labels)

# =========================
# HITUNG FREKUENSI KATA
# =========================
word_freq = {c: {} for c in classes}
class_word_count = {c: 0 for c in classes}

for tokens, label in zip(texts, labels):
    for w in tokens:
        word_freq[label][w] = word_freq[label].get(w, 0) + 1
        class_word_count[label] += 1

# =========================
# VOCABULARY
# =========================
vocab = set()
for c in word_freq:
    vocab.update(word_freq[c].keys())

vocab_size = len(vocab)

# =========================
# FUNGSI PREDIKSI
# =========================
def predict(text):
    tokens = preprocess(text)
    scores = {}

    for c in classes:
        score = math.log(prior[c])
        for w in tokens:
            count = word_freq[c].get(w, 0) + 1  # Laplace smoothing
            score += math.log(count / (class_word_count[c] + vocab_size))
        scores[c] = score

    return max(scores, key=scores.get)

# =========================
# INTERFACE
# =========================
print("Model Naive Bayes siap digunakan")

while True:
    kalimat = input("\nMasukkan kalimat (exit untuk keluar): ")
    if kalimat.lower() == "exit":
        break

    hasil = predict(kalimat)

    if hasil == 0:
        print("Sentimen: NEGATIF")
    elif hasil == 1:
        print("Sentimen: NETRAL")
    else:
        print("Sentimen: POSITIF")
