import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load('XGBoost_Tuned.pkl')
gender_encoded = joblib.load('gender_encode.pkl')
previous_loan_encoded = joblib.load('previous_loan.pkl')
education_encoded = joblib.load('education_encode.pkl')
ownership_encoded = joblib.load('ownership_encode.pkl')
intent_encoded = joblib.load('intent_encode.pkl')

def main():
    st.title('Loan Status of Customer')
    
    person_age = st.number_input('Age',min_value=0.0)
    person_gender = st.radio('Gender', ['female','male'])
    person_education = st.radio('Education',['Bachelor','Associate','High School','Master','Doctorate'])
    person_income = st.number_input('Income',min_value=0.0)
    person_emp_exp = st.number_input('Employee Experience',min_value=0,step=1)
    person_home_ownership = st.radio('Home Ownership',['RENT','MORTGAGE','OWN','OTHER'])
    loan_amnt = st.number_input('Loan Amount',min_value=0.0)
    loan_intent = st.radio('Loan Intent',['EDUCATION','MEDICAL','VENTURE','PERSONAL','DEBTCONSOLIDATION','HOMEIMPROVEMENT'])
    loan_int_rate = st.number_input('Loan Interest Rate',min_value=0.0)
    loan_percent_income = st.number_input('Loan Percent Income',min_value=0.0)
    cb_person_cred_hist_length = st.number_input('Credit History Length',min_value=0.0)
    credit_score = st.number_input('Credit Score',min_value=0,step=1)
    previous_loan_defaults_on_file = st.radio('Previous Loan',['Yes','No'])
    
    
    data = {
    'person_age' : float(person_age),
    'person_gender' : person_gender,
    'person_education' : person_education,
    'person_income' : float(person_income),
    'person_emp_exp' : int(person_emp_exp),
    'person_home_ownership' : person_home_ownership,
    'loan_amnt' :float(loan_amnt),
    'loan_intent' : loan_intent,
    'loan_int_rate' : float(loan_int_rate),
    'loan_percent_income' : float(loan_percent_income),
    'cb_person_cred_hist_length' : float(cb_person_cred_hist_length),
    'credit_score' : int(credit_score),
    'previous_loan_defaults_on_file' : previous_loan_defaults_on_file
    }
    
    df = pd.DataFrame([list(data.values())], columns=['person_age',
    'person_gender',
    'person_education',
    'person_income',
    'person_emp_exp',
    'person_home_ownership',
    'loan_amnt',
    'loan_intent',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score',
    'previous_loan_defaults_on_file']
                      )
    
    df['person_gender'] = gender_encoded.transform(df['person_gender'])
    df['previous_loan_defaults_on_file'] = previous_loan_encoded.transform(df['previous_loan_defaults_on_file'])
    
    education_df = df[['person_education']]
    ownership_df = df[['person_home_ownership']]
    intent_df = df[['loan_intent']]
    education_df = pd.DataFrame(education_encoded.transform(education_df).toarray(),columns=education_encoded.get_feature_names_out())
    ownership_df = pd.DataFrame(ownership_encoded.transform(ownership_df).toarray(),columns=ownership_encoded.get_feature_names_out())
    intent_df = pd.DataFrame(intent_encoded.transform(intent_df).toarray(),columns=intent_encoded.get_feature_names_out())

    df = pd.concat([df,education_df,ownership_df,intent_df], axis=1)

    df = df.drop(columns=['person_education','person_home_ownership','loan_intent'],axis=1)
    
    if st.button('Loan Status'):
        result = make_prediction(df)
        st.success(f"Loan Status : {result}")
        
def make_prediction(data):
    input_array = np.array(data).reshape(1,-1)
    predict_result = model.predict(input_array)
    
    return predict_result[0]

if __name__ == '__main__':
    main()