from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model_knn=pickle.load(open('Resources/Models/model_knn.pkl','rb'))
model_logit=pickle.load(open('Resources/Models/model_logit.pkl','rb'))
model_svm=pickle.load(open('Resources/Models/model_svm.pkl','rb'))
model_xgboost=pickle.load(open('Resources/Models/model_xgboost.pkl','rb'))


model_scaler=pickle.load(open('Resources/Models/scaler_model.pkl','rb'))


@app.route('/main')
def home():
    return render_template("UI.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=np.array(int_features)
    print(int_features)
    print(final)

    reshaped_data = final.reshape(1, -1)
    columns = ["department", "region", "education", "gender", "age", "length_of_service", "awards_won", "avg_training_score", "previous_year_rating", "KPIs_met_more_than_80"	]
    df = pd.DataFrame(reshaped_data, columns=columns)
    columns_dataset = ["department", "region", "education", "gender", "age", "previous_year_rating", "length_of_service", "KPIs_met_more_than_80", "awards_won", "avg_training_score"	]
    scaled_dataset = model_scaler.transform(df[columns_dataset])
    
    probability = 0
    prediction=model_knn.predict(scaled_dataset)
    probability+=prediction[0]
    prediction=model_logit.predict(scaled_dataset)
    probability+=prediction[0]
    prediction=model_svm.predict(scaled_dataset)
    probability+=prediction[0]
    prediction=model_xgboost.predict(scaled_dataset)
    probability+=prediction[0]
    
    print(probability)

    if probability==4:
        return render_template('UI.html',pred='Excellent Candidate. This employee absolutely deserves a promotion!!!!', ModelResult='{}/4 Models recommending the Employee for a Promotion'.format(probability))
    elif probability==3:
        return render_template('UI.html',pred='Strong Candidate. Highly recommend considering this employee for a promotion.', ModelResult='{}/4 Models recommending the Employee for a Promotion'.format(probability))
    elif probability==2:
        return render_template('UI.html',pred='Good Candidate. This employee has shown promise and could benefit from a promotion.', ModelResult='{}/4 Models recommending the Employee for a Promotion'.format(probability))
    elif probability==1:
        return render_template('UI.html',pred='Fair Candidate. While there is room for improvement, consider evaluating this employee for a promotion.', ModelResult='{}/4 Models recommending the Employee for a Promotion'.format(probability))
    else:
        return render_template('UI.html',pred='Not Recommended. At this point, a promotion may not be the best option for this employee.', ModelResult='{}/4 Models recommending the Employee for a Promotion'.format(probability))



if __name__ == '__main__':
    app.run(debug=True)
