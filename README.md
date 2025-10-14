This project is a simple ML project with DVC implementation

PROJECT STRUCTURE:
ml_project/
│
├── data/
│   ├── raw/                # put original CSV or JSON here
│   └── processed/          # cleaned data for training
│
├── src/
│   ├── __init__.py
│   ├── data_prep.py        # load + clean data
│   ├── train_model.py      # train model
│   ├── evaluate.py         # evaluate model
│   └── predict.py          # make predictions
│
├── requirements.txt
└── README.md

**Create a virtual env
python -m venv ml_venv
source ml_venv/bin/activate

**Command to install dependencies
pip install -r requirements.txt


**Install mlflow
pip install mlflow# ml-project

**Install dvc
pip install dvc