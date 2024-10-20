#项目的表示
'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir' WindGustSpeed', 'WindSpeed9am', ''WindSpeed3pm, 'Humidity9am',
最低温	最高温	      降雨量      蒸发量	日照        '阵风速度		风向           风速 9 am.	风速 3pm.                          湿度 9am
'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainTomorrow'
湿度 3pm		气压9 am.	   气压 3pm         云层						     风向


识别分类变量：
使用中位数填充数值列中的缺失值，并使用众数填充分类列中的缺失值
查找所有类别变量：。categorical = df.select_dtypes(include=['object']).columns.tolist()
计算二元分类变量：。binary_categorical = [col for col in categorical if df[col].nunique() == 2]

处理分类变量中的缺失值：
检查分类变量中是否存在缺失值：。missing_categorical = df[categorical].isnull().sum()
打印具有缺失值的分类变量：.print(categorical_with_missing)

识别和处理数值变量中的异常值：
使用 IQR 方法识别潜在的异常值。
绘制数值列的箱线图并识别潜在的异常值。

处理数值变量中的缺失值：
将数值列中的缺失值替换为中位数

处理分类列中的缺失值：
将分类列中的缺失值替换为模式（最常见的值）：
特征工程：

根据“RainToday”值创建二进制列“RainToday_0”和“RainToday_1”。
从数值列列表中删除目标变量“RainTomorrow”。
训练-测试拆分：

将数据集拆分为特征 （X） 和目标变量 （y）。
使用 将数据集拆分为训练集和测试集。train_test_split
目标变量的标签编码：

使用 将分类目标变量“RainTomorrow”转换为数值格式。LabelEncoder
分类特征的独热编码：

使用 对分类特征执行独热编码。pd.get_dummies
逻辑回归模型训练：

使用 在训练数据上训练逻辑回归模型。LogisticRegression

模型评估：
使用准确性在训练集和测试集上评估逻辑回归模型。
混淆矩阵和分类报告：
使用 seaborn 可视化混淆矩阵。
打印显示精确率、召回率和 F1 分数的分类报告。
受试者工作特征 （ROC） 曲线：
绘制 ROC 曲线以评估逻辑回归模型的性能。
支持向量机（SVM）分类：
使用 OneVsRestClassifier 方法训练 SVM 分类器。
使用准确性、混淆矩阵和分类报告评估 SVM 分类器。




*************************************************************************************
