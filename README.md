# Face & Expression Recognition

Three PyTorch models answering three questions about a photo, served through a single Gradio app:

| Task | Question | Model |
|---|---|---|
| Face similarity | Are these two photos the same person? | Siamese CNN → Euclidean distance |
| Face recognition | Which team member is this? | Siamese embedding → linear SVM |
| Expression recognition | What expression is this? | CNN trained from scratch (7 classes) |

Built by Group 22 as the final hackathon (Hackathon IV) of a PG Diploma programme.

## How it works

Every request goes through the same front end:

```
photo → Haar cascade face crop → grayscale + resize → model → answer
```

### 1. Siamese network (face similarity)
- Both faces pass through **one shared CNN**: 3 × (ReflectionPad → Conv 3×3 → ReLU → BatchNorm), then fully connected layers 80,000 → 500 → 500 → 5.
- Each face becomes a **5-number embedding**; the Euclidean distance between two embeddings is the dissimilarity score (low = same person).
- Trained on the AT&T faces dataset (40 people × 10 images) with **contrastive loss** (margin 2.0).

### 2. Face recognition
- The trained Siamese network is reused as a frozen feature extractor.
- A **linear SVM** (scikit-learn `SVC`, linear kernel) is trained on the embeddings of our team's photos (4 people).
- `class_mapping.json` maps the predicted index to a name.

### 3. Expression recognition
- CNN on 48×48 grayscale faces: 3 blocks of 2 × (Conv 3×3 → BatchNorm → ReLU) → MaxPool → Dropout, with 64 → 128 → 256 filters, then 9,216 → 256 → 7.
- About 3.5 M parameters; predicts Anger, Disgust, Fear, Happiness, Neutral, Sadness or Surprise.
- Trained on ~18,000 labelled face images.

## Challenges

**Docker was not available.** The provided starter app was a FastAPI service meant to run as a Docker Space on Hugging Face, but Docker Spaces were not available on our account. We wrote a thin Gradio front end (`app.py`) that calls the same prediction functions, so no model code or retraining was needed. We also pinned OpenCV below version 5, which removed the Haar cascade face detector.

**Very little Fear data.** Fear has only 456 training images, against 4,952 for Neutral and 4,760 for Happiness. A model sees Fear so rarely that it tends to predict the common classes instead, and Fear is also easily confused with Surprise and Sadness. Class-weighted loss, oversampling/augmentation of rare classes and per-class evaluation are the main remedies.

## Repository layout

```
├── app.py                  Gradio app with three tabs
├── requirements.txt
├── Hackathon_setup/
│   ├── face_recognition_model.py   Siamese network definition + transform
│   ├── face_recognition.py         get_similarity(), get_face_class()
│   ├── exp_recognition_model.py    expression CNN definition + transform
│   ├── exp_recognition.py          get_expression()
│   ├── siamese_model.t7            Siamese weights (Git LFS)
│   ├── decision_tree_model.sav     linear SVM (filename required by the template)
│   ├── class_mapping.json
│   ├── expression_model.t7         expression CNN weights (Git LFS)
│   └── haarcascade_*.xml           OpenCV face / eye detectors
└── notebooks/
    ├── 4a_face_similarity_and_recognition.ipynb
    └── 4b_expression_recognition.ipynb
```

> **Note on the notebooks:** they document an alternative training approach explored during the hackathon (a small PyTorch classifier instead of the SVM for recognition, and a 4-block expression CNN). The deployed models in `Hackathon_setup/` were trained with the architectures described above, so the notebooks will not reload those weight files directly.

## Run locally

Requires [Git LFS](https://git-lfs.com) to download the model files.

```bash
git lfs install
git clone <this-repo-url>
cd face-expression-recognition
pip install -r requirements.txt gradio
python app.py
```

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

## Tech stack

PyTorch · torchvision · OpenCV · scikit-learn · Gradio · Google Colab (GPU training)
