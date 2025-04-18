# DS440_Project

Data 1: https://www.kaggle.com/datasets/fizzbuzz/cleaned-toxic-comments?select=train_preprocessed.csv
- From Data 1, download train_preprocessed.csv (To run the Code, only this CSV file is needed)

## How to Run the Toxicity Detection App

This app uses a trained deep learning model to detect and visualize different types of text toxicity using a simple Gradio web interface.

---

### 0. Install All Required Libraries

You need to install the following Python packages before running the app.

#### Install all at once:

```bash
pip install tensorflow pandas numpy nltk matplotlib gradio
```

#### Or install one by one:

```bash
pip install tensorflow
pip install pandas
pip install numpy
pip install nltk
pip install matplotlib
pip install gradio
```

> **Important:** After installing `nltk`, download the `stopwords` resource. Open Python and run:

```python
import nltk
nltk.download('stopwords')
```

---

### 1. Train the Model (`model.py`)

Before using the interface, you must first **train and save the model**.

#### Steps:

1. Open `model.py` in **Visual Studio Code (VSCode)**.
2. Make sure `train_preprocessed.csv` is in the same folder.  
   If not, update the path inside `load_data()`.
3. Run:

```bash
python model.py
```

This will:
- Train the model.
- Save `model.h5`.
- Save `tokenizer.pkl`.

> These files are required by the interface to work.

---

### 2. Launch the Interface (`interface.py`)

After training the model:

1. Open `interface.py` in **VSCode**.
2. Run:

```bash
python interface.py
```

3. A browser window will open with the **Gradio interface**.
4. Paste a comment and select a chart type ("Bar Chart" or "Pie Chart") to see the toxicity prediction.

---

### Recommended Setup

- **Python version:** 3.8 – 3.12
- **IDE:** We recommend **Visual Studio Code (VSCode)** for running and editing the project.




