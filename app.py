from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import uuid
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid" , {"grid.color": ".9", "grid.linestyle": ":"})
sns.set_context("poster")
pio.kaleido.scope.default_format = "svg"

app = Flask(__name__)

@app.route('/', methods=["GET","POST"])
def hello_world():
    request_type_str = request.method
    if request_type_str == "GET":
        return render_template("index.html", href='static/x.png')
    else:
        text1 = request.form['text1']
        text2 = request.form['text2']
        random_string = uuid.uuid4().hex
        path = "static/" + random_string + ".png"
        # path = "static/pred_fig.png"
        model = pickle.load(open('model_pkl', 'rb'))
        np_arr = floats_str_to_np_arr(text1)
        list_heating = floats_str_to_list(text2)

        new_preds = model.predict(np_arr)
        make_picture("static/reg2021.csv", model, np_arr, list_heating, path)

        err = cal_error(new_preds, list_heating)

        return render_template('index.html', href=path, prediction_text='Result {}'.format(new_preds), prediction_text2='Error {}'.format(err))

    # test_np_input = np.array([[1],[9]])
    # # model = load("./model.joblib")
    # with open('model_pkl', 'rb') as f:
    #     model = pickle.load(f)
    # pred = model.predict(test_np_input)
    # pred_as_str = str(pred)
    # return render_template("index.html")

# @app.route('/<name>')
# def name(name):
#     return 'Hello! ' + name

def cal_error(pred,ori):
    lst = []
    for i in range(0,len(pred)):
        err = round(abs(ori[i] - pred[i])/ pred[i],2)
        lst.append(err)
    return lst

def make_picture(training_data_filename,model,new_inp_np_arr,result,output_file):

    reg = pd.read_csv(training_data_filename, parse_dates=['Timestamp'])

    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 24,
            }

    reg = reg[:80]

    plt.figure(figsize=(15, 8), dpi=150)
    plt.scatter(reg["Temperature"], reg["Heating Load"],c="grey")

    x_new = np.array(list(range(-10,15))).reshape(25,1)
    preds = model.predict(x_new)

    plt.plot(x_new.reshape(25),preds,"--", lw=2,color="salmon")

    new_preds = model.predict(new_inp_np_arr)
    plt.scatter(new_inp_np_arr.reshape(len(new_inp_np_arr)),new_preds,color="hotpink", s=200, label='Predicted')
    plt.scatter(new_inp_np_arr.reshape(len(new_inp_np_arr)),result,color="royalblue", s=200, label='Real')

    plt.xticks(fontsize=14, c="black")
    plt.yticks(fontsize=14, c="black")
    plt.xlabel('Temperature (°C)', fontdict=font)
    plt.ylabel('Heating Load (kWh)', fontdict=font)
    plt.legend(fontsize=14)

    plt.savefig(output_file)
    plt.show()

def floats_str_to_np_arr(floats_str):
    def is_float(s):
        try:
          float(s)
          return True
        except:
          return False

    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)

def floats_str_to_list(floats_str):
    def is_float(s):
        try:
          float(s)
          return True
        except:
          return False
    floats = ([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats

if __name__ == '__main__':
    app.run(debug=True)
