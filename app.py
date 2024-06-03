import joblib
from tensorflow.keras.models import load_model
from flask import Flask,request
import pandas as pd

gen_encoder=joblib.load(r'gencoder_encoder.pkl')
label_encoder=joblib.load(r'label_encoder.pkl')
bp_encoder=joblib.load(r'bp_encoder (2).pkl')
cho_encoder=joblib.load(r'cho_encoder (1).pkl')
model=load_model(r'drug_model.h5')
import numpy as np
app=Flask(__name__)
@app.route('/',methods=['POST'])

def predict():
    data=request.get_json(force=True)
    print(data)
    data=pd.DataFrame([data])
    data['Sex']=gen_encoder.transform(data['Sex'])
    data['Cholesterol']=cho_encoder.transform(data['Cholesterol'])
    data_bp=bp_encoder.transform(data[['BP']])
    data_bp=pd.DataFrame(data_bp.toarray(),columns=['BP_HIGH','BP_LOW','BP_NORMAL'])
    data=pd.concat([data,data_bp],axis='columns')
    data=data.drop('BP',axis='columns')
    print(data)
    out=model.predict(data)
    #import numpy as np
    out1=[]
    out_max=np.zeros_like(out)
    for i in range(out.shape[0]):
        out1=np.argmax(out[i])
        out_max[i][out1]=1
    print(out_max)
    output=label_encoder.inverse_transform(out_max)

    return str(output)

app.run(host='0.0.0.0',port=5010)

