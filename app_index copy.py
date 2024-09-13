from flask import Flask, render_template
import numpy as np
import pickle

# 1. ML모델을 메모리에 로딩하기
model_path = 'models/iris_model_svc.pkl'
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)



# 플라스크 객체 생성
app = Flask(__name__)

# https://http://127.0.0.1:5000/
@app.route('/', methods=['GET','POST'])
def index():
    aa = 'hello flask'
    bb = 'static/flower1.jpg'
    print(aa)
    return render_template('index.html', predict=aa, img_path=bb)
                         
if __name__ == "__main__":
    app.run(host="127.0.0.1", port="5000", debug=True)