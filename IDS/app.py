import pandas as pd
from flask import Flask, render_template, request
import joblib


app = Flask(__name__, static_url_path='', static_folder='static')
train_columns = joblib.load('define/train_columns.joblib')

# 加载预拟合的 ColumnTransformer 和训练好的模型

model = joblib.load('define/model.joblib')


def encode_input(data, train_columns):
    # 独热编码分类特征
    encoded_data = pd.get_dummies(data)

    # 添加训练期间存在但不在新数据中的缺失列
    missing_cols = set(train_columns) - set(encoded_data.columns)
    for col in missing_cols:
        encoded_data[col] = 0

    # 确保列的顺序与训练数据匹配
    encoded_data = encoded_data.reindex(columns=train_columns)

    return encoded_data


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    # 提取和处理表单数据
    form_data = request.form.to_dict()

    # 处理数值字段
    numeric_features = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
        'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
        'Cloud3pm', 'Temp9am', 'Temp3pm'
    ]
    # 数字特征名称列表
    numeric_values = {feature: float(form_data[feature]) for feature in numeric_features if feature in form_data}

    # 处理分类字段
    categorical_features = [
        # 假设这些是您数据集中的原始分类特征
        'WindGustDir', 'WindDir9am', 'WindDir3pm'
    ]  # 分类功能名称列表
    categorical_values = {feature: form_data[feature] for feature in categorical_features if feature in form_data}

    # 处理“RainToday”二进制字段
    rain_today_value = 1 if form_data.get('RainToday') == '1' else 0

    # 组合数值和分类值
    feature_values = {**numeric_values, **categorical_values, 'RainToday': rain_today_value}

    # 根据要素值创建 DataFrame
    input_data = pd.DataFrame([feature_values])

    # 使用“train_columns”对输入数据进行编码
    encoded_input_data = encode_input(input_data, train_columns)
    print("Encoded Input Data Columns:", encoded_input_data.columns)
    print("Encoded Input Data Shape:", encoded_input_data.shape)
    # 使用预处理器转换数据
    features_array = encoded_input_data

    # 进行预测
    prediction = model.predict(features_array)

    # 将结果返回给用户
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
