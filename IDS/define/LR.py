import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from joblib import dump

data = './Weather Training Data.csv'
df = pd.read_csv(data)
df.head()


##数据清理和EDA
# 查找所有分类变量
categorical = df.select_dtypes(include=['object']).columns.tolist()
# 计算二元分类变量
binary_categorical = [col for col in categorical if df[col].nunique() == 2]
# 打印分类变量列表
print(categorical)
# 打印二进制分类变量的计数和名称
print(len(binary_categorical), binary_categorical)
### 检查空值。
# 检查分类变量中的缺失值
missing_categorical = df[categorical].isnull().sum()
# 过滤缺失值的分类变量
categorical_with_missing = missing_categorical[missing_categorical > 0]
# 打印缺失值的分类变量
print(categorical_with_missing)
### 查找数值列并识别异常值
# 通过排除对象（分类）变量来查找所有数值变量
numerical = df.select_dtypes(exclude=['object']).columns.tolist()
# 打印数值变量列表
print(numerical)
# 初始化字典以存储潜在的异常值
potential_outliers = {}
# 用于识别和绘制数值列异常值的函数
for col in numerical:
    # 计算 Q1 和 Q3 值
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    # 计算 IQR
    IQR = Q3 - Q1

    # 定义潜在异常值的下限和上限
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 识别和存储潜在的异常值
    potential_outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    # 绘制列的箱线图
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")
    sns.boxplot(x=df[col], orient="h")
    plt.title(f"Boxplot of {col} with Potential Outliers")
    plt.show()

# 打印包含潜在异常值的数值列
print("Numerical Variables with Potential Outliers:")
for col, data in potential_outliers.items():
    if not data.empty:
        print(col)
# 删除异常值
numerical.remove('RainTomorrow')
for col in numerical:
    # 计算 Q1 和 Q3 值
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    # 计算 IQR
    IQR = Q3 - Q1

    # 定义潜在异常值的下限和上限
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
# 将数值列中的缺失值替换为中位数
for col in numerical:
    median_value = df[col].median()
    df[col].fillna(median_value, inplace=True)
# 将分类列中的缺失值替换为 mode（最频繁的值）
for col in categorical:
    mode_value = df[col].mode()[0]
    df[col].fillna(mode_value, inplace=True)

## 建模
# 根据 RainToday 值创建 'RainToday_0' 和 'RainToday_1' 列
df['RainToday_0'] = (df['RainToday'] == 'No').astype(int)
df['RainToday_1'] = (df['RainToday'] == 'Yes').astype(int)

### 使用 test_size = 0.1 将数据拆分为单独的训练集和测试集
X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


### 特征工程
# 从 scikit-learn 导入 LabelEncoder 类。
# 将目标变量 y_train 和 y_test 转换为 DataFrames。
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

# 使用该列的模式（最频繁值）填充 'RainTomorrow' 列中的缺失值。
y_train['RainTomorrow'].fillna(y_train['RainTomorrow'].mode()[0], inplace=True)
y_test['RainTomorrow'].fillna(y_test['RainTomorrow'].mode()[0], inplace=True)
# 创建两个 LabelEncoder 类实例，一个用于训练数据，一个用于测试数据。
train_labelled = LabelEncoder()
test_labelled = LabelEncoder()
# 将 LabelEncoder 拟合到 'RainTomorrow' 列中的唯一值
train_labelled.fit(y_train['RainTomorrow'].astype('str').drop_duplicates())
test_labelled.fit(y_test['RainTomorrow'].astype('str').drop_duplicates())
# 将原来的分类值替换为对应的数字编码。
y_train['enc'] = train_labelled.transform(y_train['RainTomorrow'].astype('str'))
y_test['enc'] = train_labelled.transform(y_test['RainTomorrow'].astype('str'))
# 从训练和测试 DataFrame 中删除原来的 'RainTomorrow' 列
y_train.drop(columns=['RainTomorrow'], inplace=True)
y_test.drop(columns=['RainTomorrow'], inplace=True)
print(y_train)
#独热处理
X_train = pd.concat([X_train[numerical], X_train[['RainToday_0', 'RainToday_1']],
                     pd.get_dummies(X_train.Location),
                     pd.get_dummies(X_train.WindGustDir),
                     pd.get_dummies(X_train.WindDir9am),
                     pd.get_dummies(X_train.WindDir3pm)], axis=1)

X_test = pd.concat([X_test[numerical], X_test[['RainToday_0', 'RainToday_1']],
                    pd.get_dummies(X_test.Location),
                    pd.get_dummies(X_test.WindGustDir),
                    pd.get_dummies(X_test.WindDir9am),
                    pd.get_dummies(X_test.WindDir3pm)], axis=1)

### 逻辑回归

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)


# 对训练数据进行预测
train_predictions = model.predict(X_train)

# 对测试数据进行预测
test_predictions = model.predict(X_test)

# 计算准确率分数
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)



### 计算混淆矩阵并使用 seaborn 热图绘制。
confusion = confusion_matrix(y_test, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False, square=True,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

## 评估
report = classification_report(y_test, test_predictions)
print(report)
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


#保存模型
# 将模型保存到文件中
dump(model, 'model.joblib')
# 将训练用的列名保存到文件中
train_columns = X_train.columns
dump(train_columns, 'train_columns.joblib')


