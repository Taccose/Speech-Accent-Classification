import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import re
from keras import layers
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
import os
from sklearn.preprocessing import LabelEncoder
import warnings
from IPython.display import HTML, display
import time
warnings.filterwarnings('ignore')


data=pd.read_csv(r"C:\Users\Tacco Feng\Documents\HW\archive\speakers_all.csv",index_col='speakerid')
data.head()
data.info()
data.drop(data.columns[8:11],axis=1,inplace=True)
data=data.fillna('NaN')
data.head()

from pathlib import Path
directory_path = r'C:\Users\Tacco Feng\Documents\HW\archive\recordings\recordings'

def feature_engineering(directory_path, data):
    p = 0

    df = pd.DataFrame()
    #tmp = pd.DataFrame()
    # p=1
    for index, row in data.iterrows():
        tmp = pd.Series()

        if os.path.isfile(directory_path+ '\\' + row['filename'] + '.mp3') == True:

          print(row['filename'])
          print(row['country'])
          tmp['filename'] = row['filename']
          tmp['country'] = row['country']
          y, sr = librosa.load(os.path.join(os.path.abspath(directory_path), row['filename'] + '.mp3'),sr=None)
          tmp['audio_vector'] = y
          tmp['samplerate'] = sr
          tmp['rms'] = np.mean(librosa.feature.rms(y=y))
          tmp['chroma_stft'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr)) #Chromagram from Short-Time Fourier Transform
          tmp['spec_cent'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)) #Spectral Centroid
          tmp['spec_bw'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)) #Spectral Bandwidth
          tmp['rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)) #Spectral Rolloff
          tmp['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y)) #Zero Crossing Rate

          mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=40) #Mel-Frequency Cepstral Coefficients
          i = 0
          for e in mfcc:
            tmp['mfcc' + str(i)] = np.mean(e)
            i += 1

          df = df._append(tmp,ignore_index=True)
          #df = pd.concat([df, tmp])
          print(p)
          p += 1
    return df

df=feature_engineering(directory_path,data)

result = pd.concat([df, df2], axis=0)
df =result

df['1stlang'] = df['filename'].apply(lambda x: re.sub(r'\d', '', x))
label_encoder = LabelEncoder()
df['accent_code'] = label_encoder.fit_transform(df['1stlang'])
condition = df['accent_code'].isin([0, 3])
df = df[condition]
numeric_columns = df.select_dtypes(include=[np.number]).columns
grouped_df = df.groupby('accent_code')[numeric_columns].mean()
replace_dict = {3:1}
# 使用 replace 方法替换 accent_code 列中的值
df['accent_code'] = df['accent_code'].replace(replace_dict)


#################################################################
#EDA
f, ((ax11, ax12)) = plt.subplots(1, 2, sharex=False, sharey=False)
# 01 左，信号
ax11.set_title('Signal')
ax11.set_xlabel('Time (samples)')
ax11.set_ylabel('Amplitude')
ax11.plot(y)
# 02 右，傅里叶变换
n_fft = 2048
ft = np.abs(librosa.stft(y[:n_fft], hop_length=n_fft + 1))
ax12.set_title('Spectrum')
ax12.set_xlabel('Frequency Bin')
# ax12.set_ylabel('Amplitude')
ax12.plot(ft)
plt.show()

n_mels = 64
n_frames = 5
n_fft = 1024
hop_length = 512
power = 2.0

mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                 sr=sr,
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 n_mels=n_mels,
                                                 power=power)
librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max),
                         y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
##################################################################################

plt.show()

# 04 将mel谱图转换为log mel谱图
log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))
librosa.display.specshow(librosa.power_to_db(log_mel_spectrogram, ref=np.max),
                         y_axis='mel', fmax=8000, x_axis='time')
# plt.colorbar(format='%+2.0f dB')
##################################################################################

plt.show()


#### MFCC BOX
# 选择需要比较的 MFCC 特征列（mfcc0 到 mfcc20）
mfcc_features = [f'mfcc{i}' for i in range(20)]  # mfcc0 到 mfcc19

# 设置子图的行数和列数
num_rows = 5  # 总共 21 个特征，每行显示 5 个，共 5 行
num_cols = 4  # 每列显示 4 个特征

# 创建子图
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))

# 遍历每个 MFCC 特征，并在相应的子图中绘制箱线图
for i, feature in enumerate(mfcc_features):
    row = i // num_cols  # 计算当前特征所在的行号
    col = i % num_cols   # 计算当前特征所在的列号
    sns.boxplot(x='1stlang', y=feature, data=df, ax=axes[row, col])
    axes[row, col].set_title(f' {feature} ')
    axes[row, col].set_xticklabels([])
    axes[row, col].set_ylabel('')
    axes[row, col].set_xlabel('')


# 调整子图之间的间距和布局
plt.tight_layout()
plt.show()



mfcc_features = [f'mfcc{i}' for i in range(20)]
# 创建子图
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 20))

# 遍历每个 MFCC 特征，并在相应的子图中绘制箱线图
for i, feature in enumerate(mfcc_features):
    row = i // num_cols  # 计算当前特征所在的行号
    col = i % num_cols   # 计算当前特征所在的列号
    sns.scatterplot(x='1stlang', y=feature, data=df, ax=axes[row, col])
    axes[row, col].set_title(f' {feature} ')
    axes[row, col].set_xticklabels([])
    axes[row, col].set_ylabel('')
    axes[row, col].set_xlabel('')


# 调整子图之间的间距和布局
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(20, 10))
for i in range(102,681): # Change diffrent group by different range
  y=df.loc[i,'audio_vector']
  n_fft = 2048
  ft = np.abs(librosa.stft(y[:n_fft], hop_length=n_fft + 1))
  plt.plot(ft, alpha=0.5)  # alpha 设置为0.5，使得每个频谱图半透明叠加

# 设置图形标题和坐标轴标签
plt.title('Stacked Spectrograms of Audio Vectors')
plt.xlabel('Time')
plt.ylabel('Frequency')

# 显示图例（可选）
# plt.legend()
plt.ylim(0, 1.5)
# 显示图形
plt.show()

df['ft'] = None

for index, row in df.iterrows():
    y = df.loc[index, 'audio_vector']
    n_fft = 2048
    ft = np.abs(librosa.stft(y[:n_fft], hop_length=n_fft + 1))
    df.at[index, 'ft'] = ft



##################################################
##Box plot 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 设置绘图风格
sns.set(style="whitegrid")

# 列出所有需要绘制箱线图的特征
features = ['rms', 'chroma_stft', 'spec_cent', 'spec_bw', 'rolloff', 'zcr']

# 创建图形对象
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

# 遍历特征并绘制箱线图
for ax, feature in zip(axes.flatten(), features):
    sns.boxplot(x='accent_code', y=feature, data=df, ax=ax)
    ax.set_title(f'Boxplot of {feature} by Group')
    ax.set_xlabel('Group')
    ax.set_ylabel(feature)

# 调整子图间距
plt.tight_layout()
plt.show()

##############################################################################################################################
##Logistic regression
import statsmodels.api as sm

df = df[df['accent_code'].isin([0, 1])]
df['intercept'] = 1.0
#'intercept', 'chroma_stft', 'spec_cent', 'spec_bw','rolloff','zcr',
X =df[[ 'rms','chroma_stft', 'spec_cent', 'spec_bw','rolloff','zcr','mfcc0', 'mfcc1', 'mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10',
        'mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19']]
y = df['accent_code']

logit_model = sm.Logit(y, X)

result = logit_model.fit()

print(result.summary())

p_values = result.pvalues
p_values_sorted = p_values.sort_values()

# 创建一个DataFrame以便绘制柱状图
p_values_df = pd.DataFrame(p_values_sorted, columns=['p_value'])

# 绘制柱状图
plt.figure(figsize=(10, 6))
p_values_df['p_value'].plot(kind='bar', color='skyblue')
plt.title('P-values of Logistic Regression Coefficients')
plt.xlabel('Features')
plt.ylabel('P-value')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=0.05, color='r', linestyle='--')  # 添加显著性水平线
plt.tight_layout()
plt.show()



##################################################################################################################
###Clusering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


######
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def plot_elbow_method(X_scaled, max_k=10):
    sse = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 7))
    plt.plot(range(1, max_k+1), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE (Sum of Squared Errors)')
    plt.title('Elbow Method for Optimal K')
    plt.show()

def perform_kmeans_with_elbow(df):
    # 选择特征
    feature_cols = [
        'rms', 'chroma_stft', 'spec_cent', 'spec_bw', 'rolloff', 'zcr',
        'mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
        'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19'
    ]
    X = df[feature_cols]

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 绘制肘部图以确定最佳K值
    plot_elbow_method(X_scaled, max_k=10)

    # 假设根据肘部图选择了最佳K值，例如2
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # PCA降维到2D进行可视化
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', marker='o')
    for i in range(len(df)):
        plt.text(X_pca[i, 0], X_pca[i, 1], str(df.index[i]), fontsize=9)
    plt.title('K-means Clustering (PCA-reduced to 2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

    return df

# 假设 df 是包含原始数据的 DataFrame
df_clustered = perform_kmeans_with_elbow(df)


def perform_clustering(df):
    # Select features for clustering
    feature_cols = [
        'rms', 'chroma_stft', 'spec_cent', 'spec_bw', 'rolloff', 'zcr',
        'mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
        'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19'
    ] #
    X = df[feature_cols]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA to reduce dimensions to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_pca)

    cluster_centers = kmeans.cluster_centers_
    # Calculate the percentage of variance explained by each of the selected components
    explained_variance_ratio = pca.explained_variance_ratio_
    total_explained_variance = np.sum(explained_variance_ratio)

    print(f"Explained variance by each component: {explained_variance_ratio}")
    print(f"Total explained variance by the 2 components: {total_explained_variance * 100:.2f}%")

    # Plot clustering result
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', marker='o')


    # Add labels to each point
    for i, txt in enumerate(df.accent_code):
        plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.7)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='X', label='Cluster Centers')
    plt.title('K-means Clustering of Audio Features (PCA-reduced)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.show()

    silhouette_avg = silhouette_score(X_scaled, df['cluster'])
    print(f"Silhouette Score: {silhouette_avg}")
    return df

df_clustered = perform_clustering(df)

##3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.metrics import silhouette_score

def perform_clustering(df):
    # Select features for clustering
    feature_cols = [
        'rms', 'chroma_stft', 'spec_cent', 'spec_bw', 'rolloff', 'zcr',
        'mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
        'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19'
    ]
    X = df[feature_cols]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA to reduce dimensions to 3D for visualization
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_pca)

    # Calculate the percentage of variance explained by each of the selected components
    explained_variance_ratio = pca.explained_variance_ratio_
    total_explained_variance = np.sum(explained_variance_ratio)

    print(f"Explained variance by each component: {explained_variance_ratio}")
    print(f"Total explained variance by the 3 components: {total_explained_variance * 100:.2f}%")

    # Plot clustering result in 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=df['cluster'], cmap='viridis', marker='o')

    cluster_centers = kmeans.cluster_centers_
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], c='red', s=200, marker='X', label='Cluster Centers')

    for i, txt in enumerate(df.accent_code):
        ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], txt, fontsize=8, alpha=0.7)
       # plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]), fontsize=8, alpha=0.7)

    # Adding labels and title
    ax.set_title('K-means Clustering of Audio Features (PCA-reduced to 3D)')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    fig.colorbar(scatter, label='Cluster')
    plt.show()

    silhouette_avg = silhouette_score(X_scaled, df['cluster'])
    print(f"Silhouette Score: {silhouette_avg}")

    return df


df_clustered = perform_clustering(df)


##############################################################################################
##Ramdon Forrest with cluster labels
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
def evaluate_feature_importance(df):
    # Select features and target
    feature_cols = [
        'rms', 'chroma_stft', 'spec_cent', 'spec_bw', 'rolloff', 'zcr',
        'mfcc0', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
        'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19'
    ]
    X = df[feature_cols]
    y = df['cluster']  # Use cluster labels as the target for simplicity

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)

    # Get feature importances
    feature_importances = pd.Series(rf.feature_importances_, index=feature_cols)
    feature_importances = feature_importances.sort_values(ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 7))
    feature_importances.plot(kind='bar')
    plt.title('Feature Importances from Random Forest')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.show()

    return feature_importances


# Evaluate feature importance
feature_importances = evaluate_feature_importance(df_clustered)
print("Feature Importances:\n", feature_importances)



###########################################################################################################
###Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

#'chroma_stft', 'spec_cent', 'spec_bw','rolloff','zcr',
X =df[[ 'rms','chroma_stft', 'spec_cent', 'spec_bw','rolloff','zcr','mfcc0', 'mfcc1', 'mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10',
        'mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19']]
y = df['accent_code']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lasso = Lasso(alpha=0.1)  # 设置 Lasso 回归的 alpha 参数
lasso.fit(X_train_scaled, y_train)


print("Lasso：")
for feature, coef in zip(X.columns, lasso.coef_):
    print(f"{feature}: {coef}")


y_pred = lasso.predict(X_test_scaled)


print("Lasso Report：")
print(classification_report(y_test, y_pred.round()))

dd = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
latex_table = dd.to_latex(index=False)
print(latex_table)



###########################################################################################################
###Ramdon forest

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

#
X =df[[ 'rms','chroma_stft', 'spec_cent', 'spec_bw','rolloff','zcr','mfcc0', 'mfcc1', 'mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10',
        'mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19']]
#X =df[[ 'spec_cent', 'spec_bw','rolloff','mfcc0', 'mfcc1', 'mfcc3','mfcc5','mfcc6','mfcc7','mfcc9','mfcc10',
 #       'mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19']]
y = df['accent_code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)  # 设置随机森林参数
random_forest.fit(X_train_scaled, y_train)

y_pred = random_forest.predict(X_test_scaled)

print("RF report：")
print(classification_report(y_test, y_pred))

print(random_forest.get_params())

# 提取特征重要性
importances = random_forest.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(feature_importance_df)

# 绘制特征重要性图
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance from Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

##################################################################################################################
#Cat boost
import pandas as pd
import numpy as np
import librosa
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
#'chroma_stft', 'spec_cent', 'spec_bw','rolloff','zcr',
X =df[['mfcc0', 'mfcc1', 'mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10','mfcc11',
    'mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19']]


X =df[[ 'chroma_stft', 'spec_cent', 'spec_bw', 'rolloff', 'zcr',
         'mfcc1',  'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7',
        'mfcc10', 'mfcc12', 'mfcc13', 'mfcc14',  'mfcc16', 'mfcc17', 'mfcc18']]
y = df['accent_code']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练CatBoost模型
catboost_model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, verbose=100)
catboost_model.fit(X_train_scaled, y_train)

# 预测并打印分类报告
y_pred = catboost_model.predict(X_test_scaled)
print("CatBoost Classification Report:")
print(classification_report(y_test, y_pred))

# 提取特征重要性
feature_importance = catboost_model.get_feature_importance(Pool(X_train_scaled, label=y_train))
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(feature_importance_df)

# 绘制特征重要性图
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance from CatBoost Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

####################################################################################
###CNN
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5120, 128)
        # self.fc1 = nn.Linear(22016, 128)
        # self.fc1 = nn.Linear(5504, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

### Transform
class TransformerModel(nn.Module):
    def __init__(self, num_classes, nhead=8, num_encoder_layers=3, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Transformer expects [seq_len, batch, features]
        transformed = self.transformer_encoder(src)
        output = transformed.mean(dim=0)
        output = self.dropout(output)
        output = self.fc_out(output)
        return output

###Resample
class AudioDataset(Dataset):
    def __init__(self, data_path, transform=None, target_sample_num=None):
        self.data_path = data_path
        self.transform = transform
        self.audios, self.labels = self.load_audios()
        if target_sample_num is not None:
            self.audios, self.labels = self.rebalance(self.audios, self.labels, target_sample_num)

    def load_audios(self):
        audios, labels = [], []
        for label in os.listdir(self.data_path):
            speaker_path = os.path.join(self.data_path, label)
            for filename in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, filename)
                audio, sr = librosa.load(file_path, sr=None)
                audio = librosa.util.fix_length(audio, size=22050*8)  # Fix all audios to 2/8 seconds
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
                log_mel_spec = librosa.power_to_db(mel_spec)
                audios.append(log_mel_spec)
                labels.append(label2id[label])
        with open('audio_data_8.pkl', 'wb') as f:
            pickle.dump((audios, labels), f)

        with open('audio_data_8.pkl', 'rb') as f:
            audios, labels = pickle.load(f)
        return audios, labels

    def rebalance(self, audios, labels, target_num):
        label_counter = Counter(labels)
        new_audios, new_labels = [], []
        for label in set(labels):
            label_samples = [(audio, lab) for audio, lab in zip(audios, labels) if lab == label]
            current_sample_num = len(label_samples)
            if current_sample_num < target_num:
                repeats = (target_num // current_sample_num) + 1
                label_samples = (label_samples * repeats)[:target_num]
            else:
                label_samples = random.sample(label_samples, target_num)
            new_audios.extend([audio for audio, _ in label_samples])
            new_labels.extend([label for _, label in label_samples])
        return new_audios, new_labels

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        audio = self.audios[idx]
        label = self.labels[idx]
        if self.transform:
            audio = self.transform(audio)
        return audio, label
        
        
def load_data(data_path, resample=False):
    if resample:
        dataset = AudioDataset(data_path, target_sample_num=600)
    else:
        dataset = AudioDataset(data_path)
    le = LabelEncoder()
    labels_encoded = le.fit_transform(dataset.labels)
    labels_encoded = one_hot(torch.tensor(labels_encoded), num_classes=len(le.classes_))
    train_idx, test_idx = train_test_split(range(len(labels_encoded)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader, len(le.classes_)

### Train and test
def train_model(model, train_loader, criterion, optimizer, epochs=50, is_transformer=None):
    train_losses, valid_accuracies = [], []
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            if not is_transformer:
                inputs = inputs.unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
        valid_accuracie = evaluate_model(model, test_loader, is_transformer)
        train_losses.append(running_loss / len(train_loader))
        valid_accuracies.append(valid_accuracie)



    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracies, label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_loader, is_transformer):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            if not is_transformer:
                inputs = inputs.unsqueeze(1).float()
            # labels = labels.argmax(dim=1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}%')
    return 100 * correct / total

### Main
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

labels_all = ['english',  'arabic', 'spanish', 'french', 'mandarin', 'others']  # label -> id
label2id, id2label = dict(), dict()
for i, label in enumerate(labels_all):
    label2id[label] = i
    id2label[i] = label
print(id2label)

# Resampling is very important
train_loader, test_loader, num_classes = load_data('archive/recordings', resample=True)

is_transformer = False
if is_transformer:
    model = TransformerModel(num_classes=num_classes).to(device)
else:
    model = CNN(num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
train_model(model, train_loader, criterion, optimizer, epochs=num_epochs, is_transformer=is_transformer)
# evaluate_model(model, test_loader)

'''
The dataset should be preprocessed into the following structure:
archive
├── recordings/
│   ├── arabic
│   ├── english
|   ├── french
|   ├── mandarin
|   ├── others
|   ├── spanish
├── speakers_all.csv
'''
##################################
