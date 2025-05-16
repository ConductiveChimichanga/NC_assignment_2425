# Food Classification with CNN - Assignment fot the Neural Computing Course Leiden DSAI

This project implements a deep learning-based food image classification system using a custom Convolutional Neural Network (CNN) in PyTorch. The goal is to classify food images into 91 categories and use these predictions to infer user taste profiles for restaurant recommendations.

## Contents

- Jupyter Notebook: Full code and documentation for model development and training.
- README (this file): Instructions and requirements.
- Report: Summary of approach and results.

---

## Group Information

- **Group Number:** 54
- **Member Names and sIDs:**
  -  Alma Rosenmann (s4078810)
  - Matei Canavea (s3930025)
  - Noam Dunsky (s4009495)
  - Peter van der Steeg (s3312259)

---
## Installing requirements

### Via requirements.txt and pip 
1. Create a `requirements.txt` file with the following content:

    ```
    python==3.11.11
    torch==2.0.1
    torchvision==0.15.2
    numpy==1.26.4
    tqdm==4.66.2
    ```

2. Install the dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

> Make sure to adjust the Python version if you're using a virtual environment manager like `venv` or `conda` that already defines the Python version.

### Via pip/conda directly

#### Using `pip`:

```bash
pip install torch==2.0.1 torchvision==0.15.2 numpy==1.26.4 tqdm==4.66.2
```

#### Using conda

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 numpy==1.26.4 tqdm==4.66.2 -c pytorch
```

## Running `NC_groupXX.ipynb`

Follow the steps below to open and run the notebook in either **Jupyter Notebook**, **JupyterLab**, or **Visual Studio Code**.

### 1. Open the Notebook

- **Using Jupyter Notebook or JupyterLab:**
  - Launch Jupyter Notebook or JupyterLab.
  - Navigate to the directory containing `NC_groupXX.ipynb`.
  - Click on the notebook to open it.

- **Using Visual Studio Code (VS Code):**
  - Open Visual Studio Code.
  - Open the folder containing `NC_groupXX.ipynb`.
  - Click on the notebook file to open it in the interactive editor.
  - Make sure the **Python extension** is installed and active.

### 2. Execute the Cells

- Run each cell **sequentially from top to bottom**.
- In both Jupyter and VS Code:
  - Press `Shift + Enter` to run the current cell and move to the next.
  - Alternatively, in VS Code, you can also click the **Run Cell** button.

> Make sure that all dependencies are installed and the correct Python environment is selected before running the notebook.

---

## Dataset

- The dataset must be placed in the following structure:
  ```
  /data/
        train/
        test/
  ```
- Each class should be in its own subdirectory.
- **Do not include the dataset in your submission.**

---

## Reproducibility

- The code sets a fixed random seed and configures PyTorch/CUDA for deterministic results.
- All hyperparameters and model architecture details are specified in the notebook.

---

## Additional Information

- All hyperparameters (batch size, learning rate, epochs, etc.) are defined at the top of the notebook.
- The model is trained from scratch (no pretrained weights).
- The code reports test accuracy after every epoch and saves the best model as `best_model.pth`.

  **Notes:**
  - If running on Kaggle or another environment, match the versions above for reproducibility.
  - No other external libraries are required.


---

**End of README**
