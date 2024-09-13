from flask import Flask, render_template, request
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

    # 사용자 입력 데이터를 받아서 에측수행
    ## 클라이언트에 넘어온 요청이 post일 경우
    if request.method == 'POST':

        ## 4개의 데이터를 입력 value를 뽑고 -> numpy 2d구조로 만들고
        #print(type(request.form['sepal_length']))
        #print(request.form['sepal_width'])
        #print(request.form['petal_length'])
        #print(request.form['petal_width'])

        sl = request.form['sepal_length']
        sw = request.form['sepal_width']
        pl = request.form['petal_length']
        pw = request.form['petal_width']

        # numpy 2d로 만들기
        input_data = np.array([[float(sl), float(sw), float(pl), float(pw)]])

        ## 모델 예측 수행 -> 결과 index.html에 렌더링
        prediction = model.predict(input_data)
        predicted_label = ['setosa', 'versicolor', 'virginica'][prediction[0]]
        if predicted_label == 'setosa':
            img_path = 'static/setosa.jpg'
        elif predicted_label == 'versicolor':
            img_path = 'static/versicolor.jpg'
        else:
            img_path = 'static/virginica.png'
        return render_template('index.html', predict=predicted_label, img_path=img_path)

    # 데이터 입력 페이지 처리 (get방식)
    return render_template('index.html')

                         
if __name__ == "__main__":
    app.run(host="127.0.0.1", port="5000", debug=True)