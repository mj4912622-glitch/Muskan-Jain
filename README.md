README
Pascal Vehicle Classification (Decision Tree / Random Forest / SVM)

Short description:
This repository contains three Jupyter notebooks that implement classical machine-learning pipelines for vehicle classification using the PASCAL dataset (feature extraction → feature engineering → model training → evaluation). The notebooks show data loading, preprocessing, feature extraction, model training (Decision Tree, Random Forest, SVM), and common evaluation plots (confusion matrix, classification report, accuracy curves).

Contents
Pascal_DT+BE.ipynb         # Decision Tree pipeline (with the feature-engineering step labeled "BE")
Pascal_RF+BE.ipynb         # Random Forest pipeline (with the feature-engineering step labeled "BE")
Pascal_SVM+BE.ipynb        # SVM pipeline (with the feature-engineering step labeled "BE")
README.md                  # (this file)


Note: The notebooks refer to a feature-engineering step abbreviated BE (see the notebooks for the exact implementation and explanations of that step).

Key features

Data loading from PASCAL (VOC) formatted images/annotations (not included in repo).

Image preprocessing and feature extraction (HOG and other features / BE step implemented inside notebooks).

Model experiments:

Decision Tree

Random Forest

Support Vector Machine (SVM)

Train / test split, cross-validation option (where implemented).

Evaluation: accuracy, precision/recall/F1 (classification report), confusion matrix, and example visualizations.

Save / load model checkpoint examples (where applicable inside notebooks).

Requirements

Create a Python environment (recommended: venv or conda) and install the dependencies used in the notebooks:

python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install --upgrade pip
pip install jupyterlab numpy scipy scikit-learn matplotlib seaborn opencv-python scikit-image pandas joblib tqdm


(If a notebook uses extra libraries, the notebook header cells include the full import list — install any additional packages as required.)

How to run

Clone this repository (or copy the notebooks to your working folder).

Prepare the dataset:

Download and extract the PASCAL VOC dataset (or your chosen dataset).

Update dataset path variables in each notebook (search for DATA_DIR / data_dir).

Start Jupyter and open the notebook you want to run:

jupyter lab
# or
jupyter notebook


Run the notebook cells in order. Each notebook is written as a self-contained experiment and provides visualizations and printed evaluation metrics.

Run notebooks headless (optional)

To execute a notebook from shell and save output:

# requires nbconvert
pip install nbconvert
jupyter nbconvert --to notebook --execute Pascal_RF+BE.ipynb --output executed_Pascal_RF+BE.ipynb

Typical workflow inside each notebook

Configuration / paths — set dataset path, model output path, parameters.

Data loading — read images and labels (PASCAL VOC parsing if implemented).

Preprocessing — resizing, gray scale conversion, normalization.

Feature extraction — HOG (example), and the BE feature engineering block (see notebook for details).

Model training — train Decision Tree / Random Forest / SVM with hyperparameters.

Evaluation — classification report, confusion matrix, and optional cross-validation.

Model export — save trained model with joblib or pickle.

Results & Notes

Each notebook prints model performance (accuracy, precision/recall/F1) and displays a confusion matrix.

For fair comparisons, ensure the same preprocessing/feature steps and the same train/test splits are used across notebooks.

If you want to extend:

Add cross-validation and hyperparameter search (GridSearchCV / RandomizedSearchCV).

Add additional features or deep-features (pretrained CNN embeddings).

Try ensemble methods or stacking.

Reproducibility tips

Fix the random seed where relevant (notebooks include random_state).

Make sure image resizing & HOG parameters are identical across experiments when comparing models.

If dataset contains class imbalance, consider stratified splitting or class weights.

Cite / Data source

If you use PASCAL VOC dataset in your work, cite/acknowledge the dataset per its terms. For more information about the Pascal VOC challenge, see the official resources for PASCAL VOC (search “PASCAL VOC dataset” for the correct citation and download link).

Troubleshooting

cv2 import errors: install the opencv-python package or use a conda channel with OpenCV.

Running out of memory: reduce batch_size or image size in preprocessing.

Slow training: try using fewer features or downsampling the dataset while debugging.

License

This repository is provided under the MIT License. See LICENSE file (add one if you want).
If you want a different license (GPL, Apache, etc.), replace the license file accordingly.

Contributing

Contributions are welcome: open an issue to discuss improvements (e.g., add hyperparameter search, unify preprocessing into a script, add Dockerfile / environment.yml). When submitting PRs, include reproducible instructions and, if possible, a small test dataset.

Contact

If you want me to convert these notebooks into:

a single consolidated script (train.py / evaluate.py),

a README with more precise explanations extracted from the notebooks,

or a requirements.txt / environment.yml / Dockerfile — tell me which one and I’ll prepare it.
