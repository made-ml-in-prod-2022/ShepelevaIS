Pipeline is tested on the dataset from kaggle:
https://www.kaggle.com/datasets/shubh0799/churn-modelling


The goal is to predict churn of the customers from the bank.
Features:
* CustomerId - unique customer identifier
* Surname - surname
* CreditScore - credit rating
* Geography - country of residence
* Gender - gender
* Age - age
* Tenure - how many years a person has been a client of the bank
* Balance - account balance
* NumOfProducts - the number of bank products used by the client
* HasCrCard - availability of a credit card
* IsActiveMember - client activity
* EstimatedSalary - estimated salary

Target column:
* Exited - the fact of the client's departure

Environment creation
conda create --name mlprod python=3.9 pip
conda activate mlprod
python -m ipykernel install --user --name mlprod
pip install -r requirements.txt

Running scripts
python src/train_pipeline.py configs/train_config.yml

Running tests
pytest tests/