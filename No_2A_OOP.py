import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle as pkl

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.data = None
        self.target = None
        self.categorical = []
        self.numerical = []

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        
    def detect_columns(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.categorical.append(col)
            else:
                self.numerical.append(col)
        print("Categorical Columns:", self.categorical)
        print("Numerical Columns:", self.numerical)
        
    def clean_anomaly(self):
        self.df['person_gender'].replace('Male', 'male', inplace=True)
        self.df['person_gender'].replace('fe male', 'female', inplace=True)
        self.df['person_income'].fillna(self.df['person_income'].mean(), inplace=True)
        self.df['person_age'] = self.df['person_age'].astype('int64')
        self.df['cb_person_cred_hist_length'] = self.df['cb_person_cred_hist_length'].astype('int64')
        
    def create_X_y(self, target_column):
        self.target = self.df[target_column]
        self.data = self.df.drop(target_column, axis=1)
        
class ModelHandler:
    def __init__(self,target,data):
        self.target = target
        self.data = data
        self.x_train = [None]
        self.x_test = [None]
        self.y_train = [None]
        self.y_test = [None]
        self.model = [None]
        self.x_train_enc = [None]
        self.x_test_enc = [None]
        self.xgb_best = [None]
        self.y_pred = [None]
        self.createModel()
        
    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=0.3, random_state=0)
    
    def createModel(self):
        self.model = XGBClassifier()
        
    def label_encode_columns(self):
        gender_encoded = LabelEncoder()
        previous_loan_encoded = LabelEncoder()

        self.x_train['person_gender'] = gender_encoded.fit_transform(self.x_train['person_gender'])
        self.x_test['person_gender'] = gender_encoded.transform(self.x_test['person_gender'])
        self.x_train['previous_loan_defaults_on_file'] = previous_loan_encoded.fit_transform(self.x_train['previous_loan_defaults_on_file'])
        self.x_test['previous_loan_defaults_on_file'] = previous_loan_encoded.transform(self.x_test['previous_loan_defaults_on_file'])
        
        pkl.dump(gender_encoded,open('gender_encode.pkl','wb'))
        pkl.dump(previous_loan_encoded,open('previous_loan.pkl','wb'))
        
    def one_hot_encode_columns(self):
        education_train = self.x_train[['person_education']]
        ownership_train = self.x_train[['person_home_ownership']]
        intent_train = self.x_train[['loan_intent']]

        education_test = self.x_test[['person_education']]
        ownership_test = self.x_test[['person_home_ownership']]
        intent_test = self.x_test[['loan_intent']]

        train_encoded_education=OneHotEncoder()
        train_encoded_ownership=OneHotEncoder()
        train_encoded_intent=OneHotEncoder()

        education_train = pd.DataFrame(train_encoded_education.fit_transform(education_train).toarray(),columns=train_encoded_education.get_feature_names_out())
        ownership_train = pd.DataFrame(train_encoded_ownership.fit_transform(ownership_train).toarray(),columns=train_encoded_ownership.get_feature_names_out())
        intent_train = pd.DataFrame(train_encoded_intent.fit_transform(intent_train).toarray(),columns=train_encoded_intent.get_feature_names_out())

        education_test = pd.DataFrame(train_encoded_education.transform(education_test).toarray(),columns=train_encoded_education.get_feature_names_out())
        ownership_test = pd.DataFrame(train_encoded_ownership.transform(ownership_test).toarray(),columns=train_encoded_ownership.get_feature_names_out())
        intent_test = pd.DataFrame(train_encoded_intent.transform(intent_test).toarray(),columns=train_encoded_intent.get_feature_names_out())

        self.x_train = self.x_train.reset_index(drop=True)
        self.x_test = self.x_test.reset_index(drop=True)

        self.x_train_enc = pd.concat([self.x_train,education_train,ownership_train,intent_train], axis=1)
        self.x_test_enc = pd.concat([self.x_test,education_test,ownership_test,intent_test], axis=1)
        
        self.x_train_enc = self.x_train_enc.drop(columns=['person_education','person_home_ownership','loan_intent'],axis=1)
        self.x_test_enc = self.x_test_enc.drop(columns=['person_education','person_home_ownership','loan_intent'],axis=1)

        pkl.dump(train_encoded_education,open('education_encode.pkl','wb'))
        pkl.dump(train_encoded_ownership,open('ownership_encode.pkl','wb'))
        pkl.dump(train_encoded_intent,open('intent_encode.pkl','wb'))
        
        
    def train_with_gridsearch(self):
        xgb_params = {
        'max_depth': [12, 24, 36, 48],
        'learning_rate': [0.001, 0.01, 0.1, 0.2],
        'n_estimators': [150, 200, 250, 300]
        }
    
        self.model = GridSearchCV(self.model,
                        param_grid=xgb_params,
                        scoring='accuracy',
                        cv=5
                        )
        self.model.fit(self.x_train_enc,self.y_train)

        print("Tuned Hyperparameters :", self.model.best_params_)
        print("Accuracy :",self.model.best_score_)
        
        best_params = self.model.best_params_
        self.xgb_best = XGBClassifier(**best_params)
        self.xgb_best.fit(self.x_train_enc, self.y_train)

    def make_prediction(self):
        self.y_pred = self.xgb_best.predict(self.x_test_enc)
    
    def evaluate_model(self):
        print("Classification Report:\n")
        print(classification_report(self.y_test, self.y_pred))

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pkl.dump(self.model, f)
        

file_path = 'Dataset_A_loan.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.detect_columns()
data_handler.clean_anomaly()
data_handler.create_X_y('loan_status')

X = data_handler.data
y = data_handler.target

model_handler = ModelHandler(y,X)
model_handler.split_data()
model_handler.label_encode_columns()
model_handler.one_hot_encode_columns()
model_handler.train_with_gridsearch()
model_handler.make_prediction()
model_handler.evaluate_model()
model_handler.save_model('XGBoost_OOP.pkl')