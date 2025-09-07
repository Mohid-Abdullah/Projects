This is a **practice machine learning project** built with Python. The goal is to explore a simple ML workflow for sales prediction using the **Sample - Superstore** dataset.

**Note:** The dataset is not ideal for prediction, so model performance (R² score and MSE) is poor. The focus here is on **learning project structure, preprocessing, and model training** rather than achieving high accuracy.

---

## Project Structure

```
.
├── artifacts/              # Stores trained models, evaluation metrics, etc.  
├── Cleaned/                # Preprocessed data saved here  
├── __pycache__/            # Auto-generated Python cache files  
├── Data_Class              # (Placeholder for future class definitions)  
├── data_lib.py             # Helper functions for data handling  
├── Data_PP.py              # Preprocessing script (cleans and saves data to Cleaned/)  
├── main.py                 # Main script (loads cleaned data, trains RandomForest, evaluates)  
├── Sample - Superstore.csv # Original dataset  
```

---

## Workflow

1. **Dataset**

   * `Sample - Superstore.csv` → Original raw dataset.

2. **Helper Functions (`data_lib.py`)**

   * Contains reusable functions to simplify data handling.

3. **Preprocessing (`Data_PP.py`)**

   * Cleans the raw dataset.
   * Handles missing values, transformations, and encodings.
   * Saves the processed dataset into the `Cleaned/` folder.

4. **Model Training (`main.py`)**

   * Loads the cleaned dataset.
   * Splits data into train/test sets.
   * Trains a **RandomForestRegressor** to predict sales.
   * Evaluates with **R² score** and **MSE** (results are poor due to dataset quality).

---

## How to Run

1. Clone or download the project.

2. Install dependencies:

   ```bash
   pip install sklearn pandas numpy
   ```

3. Run preprocessing:

   ```bash
   python Data_PP.py
   ```

4. Train and evaluate model:

   ```bash
   python main.py
   ```

---

The project demonstrates the **pipeline of ML projects**:

* Data preprocessing
* Training
* Evaluation
* Saving artifacts

---

## Purpose

This project is meant for **practice and learning**, not production. It shows how to structure code for:

* Modular preprocessing
* Clean separation of logic (helper functions, preprocessing, main script)
* Training a regression model

---

