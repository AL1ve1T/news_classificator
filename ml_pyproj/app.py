#
#   File contains entrypoint for application
#

import csv
from dataset import Dataset
from model import Model
from flask import Flask,render_template,url_for,request

app = Flask(__name__)

with open('./uci-news-aggregator.csv', 'rt') as f:
    data = csv.reader(f)
    data = list(data)[:100000]

dataset = Dataset(data)
dataset.prepare_for_cats(['b', 'm', 't', 'e'])
model = Model(dataset)
model.train()
print('Now try out your trained model!')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process', methods=["POST"])
def process():
    if request.method == 'POST':
        text = request.form['rawtext']
        answer = model.test(text)
        return render_template("index.html", Category=answer)

def not_found(e):
    print(e)


app.run(debug=True)
