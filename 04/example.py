import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

from downloadfile import download_pubg_data

DATA_PATH = "data/pubg-stats-dataset/Pubg_Stats.csv"

## PUBG 데이터를 불러오는데 있으면 그대로 불러오고 없으면 다운로드한다.
## Kaggle의 임시 아이디/비밀번호는 아무거나 쳐도 상관없다.

try:
    ori_data = pd.read_csv(DATA_PATH)
    if ori_data is None or ori_data.empty:
        raise ValueError("Empty dataframe")
except (FileNotFoundError, ValueError):
    download_pubg_data()
    ori_data = pd.read_csv(DATA_PATH)

results = {}

print(f"---------------------------")
print("기본 배틀그라운드 데이터: ")
print(ori_data.head())
print()

# 점수는 배틀그라운에서 오래해본 내가 중요도로 생각하는 가중치를 각자 임시로 주었다
ori_data['Score'] = (
    ori_data['Kills'] * 5 +
    ori_data['Damage_Dealt'] +
    ori_data['Wins'] * 10 +
    ori_data['Top_10s'] * 8
)

def plot_scatter(data, title):
    for rank in data['Rank'].unique():
        temp = data[data['Rank'] == rank]
        plt.scatter(temp['Score'], temp['Matches_Played'], label=rank, s=20, alpha=0.7)
    plt.xlabel("Score")
    plt.ylabel("Matches Played")
    plt.title(title)
    plt.legend()
    plt.show()

plot_scatter(ori_data, "Original Data")

# 소규모 데이터에서도 효과적이고 이상치에 강한 성능을 보이기 때문에
# SVM 알고리즘을 사용하여 모델 학습을 진행

# 아래의 코드는 데이터가 제대로 정제되지 않았기 때문에 데이터가 아예 처리가 안되는 모습을 확인할 수 있다.
# 그래서 주석을 풀고 실행하면 에러가 발생하는 것을 알 수 있다.
# X = ori_data.drop(columns=['Rank'])
# y = ori_data['Rank']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# svm_model = SVC(kernel='linear')
# svm_model.fit(X_train, y_train)

# accuracy = svm_model.score(X_test, y_test)
# print(f"---------------------------")
# print(f"데이터 전처리 BEFORE :")
# print(f"정확도 : {accuracy:.4f}")

# sample = X_test.iloc[0].values.reshape(1, -1)
# prediction = svm_model.predict(sample)
# predicted_rank = prediction[0]

# print(f"예측된 랭크 : {predicted_rank}")
# print()

## 데이터 전처리 결측치 제거
# 우선 불필요한 데이터를 제거하자.
df = ori_data.drop(columns=['Player_Name', 'Unnamed: 0'])
df = df.dropna()

print(f"---------------------------")
print("불필요 데이터 제거")
print(df.head())
print()

plot_scatter(df, "After Remove N/A")

X = df.drop(columns=['Rank'])
y = df['Rank']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

accuracy = svm_model.score(X_test, y_test)
print(f"---------------------------")
print(f"데이터 전처리 결측치 제거 :")
print(f"정확도 : {accuracy:.4f}")
results['Without N/A Accuracy'] = accuracy

sample = X_test.iloc[0].values.reshape(1, -1)
prediction = svm_model.predict(sample)
predicted_rank = prediction[0]

print(f"예측된 랭크 : {predicted_rank}")
print()

## 데이터 전처리 정규화
df = ori_data.drop(columns=['Player_Name', 'Unnamed: 0'])
df = df.dropna()

X = df.drop(columns=['Rank'])
y = df['Rank']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# 데이터 정규화 (MinMaxScaler: 0~1 범위로 스케일)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

accuracy = svm_model.score(X_test, y_test)
print(f"---------------------------")
print(f"데이터 전처리 정규화 (MinMaxScaler) :")
print(f"정확도 : {accuracy:.4f}")
results['MinMaxScaler Accuracy'] = accuracy

sample = X_test[0].reshape(1, -1)
prediction = svm_model.predict(sample)
predicted_rank = prediction[0]

print(f"예측된 랭크 : {predicted_rank}")
print()

## 데이터 전처리 표준화
df = ori_data.drop(columns=['Player_Name', 'Unnamed: 0'])
df = df.dropna()

X = df.drop(columns=['Rank'])
y = df['Rank']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

# 데이터 표준화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

accuracy = svm_model.score(X_test, y_test)
print(f"---------------------------")
print(f"데이터 전처리 표준화 (StandardScaler) :")
print(f"정확도 : {accuracy:.4f}")
results['StandardScaler Accuracy'] = accuracy

sample = X_test[0].reshape(1, -1)
prediction = svm_model.predict(sample)
predicted_rank = prediction[0]

print(f"예측된 랭크 : {predicted_rank}")
print()

## 데이터 전처리 이상치 제거 (클래스별 IQR)
# 전체 기준 IQR은 Silver(7개) 전체를 이상치로 판정해 삭제하므로 클래스별로 적용
df = ori_data.drop(columns=['Player_Name', 'Unnamed: 0'])
df = df.dropna()

numeric_cols = df.select_dtypes(include=[np.number]).columns
clean_parts = []
for rank, group in df.groupby('Rank'):
    Q1 = group[numeric_cols].quantile(0.25)
    Q3 = group[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((group[numeric_cols] < (Q1 - 1.5 * IQR)) | (group[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    clean_parts.append(group[mask])
df = pd.concat(clean_parts)

print(f"---------------------------")
print(f"이상치 제거 후 데이터 크기: {df.shape[0]}행 (원본: {ori_data.shape[0]}행)")
print(f"클래스별 분포: {df['Rank'].value_counts().to_dict()}")
print()

plot_scatter(df, "After Remove Outliner")

X = df.drop(columns=['Rank'])
y = df['Rank']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 이상치 제거 + 정규화
scaler = MinMaxScaler()
X_train_mm = scaler.fit_transform(X_train)
X_test_mm = scaler.transform(X_test)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_mm, y_train)

accuracy = svm_model.score(X_test_mm, y_test)
print(f"---------------------------")
print(f"데이터 전처리 이상치 제거 + 정규화 :")
print(f"정확도 : {accuracy:.4f}")
results['MinMaxScaler + Outliner X Accuracy'] = accuracy

sample = X_test_mm[0].reshape(1, -1)
prediction = svm_model.predict(sample)
print(f"예측된 랭크 : {prediction[0]}")
print()

# 이상치 제거 + 표준화
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_std, y_train)

accuracy = svm_model.score(X_test_std, y_test)
print(f"---------------------------")
print(f"데이터 전처리 이상치 제거 + 표준화 :")
print(f"정확도 : {accuracy:.4f}")
results['StandardScaler + Outliner X Accuracy'] = accuracy

sample = X_test_std[0].reshape(1, -1)
prediction = svm_model.predict(sample)
print(f"예측된 랭크 : {prediction[0]}")
print()

## 데이터 증강 (SMOTE)
augdf = ori_data.drop(columns=['Player_Name', 'Unnamed: 0'])
augdf = augdf.dropna()

rank_categories = augdf['Rank'].astype('category')
rank_map = dict(enumerate(rank_categories.cat.categories))  # 숫자 -> 랭크명 역매핑
augdf = augdf.copy()
augdf['Rank'] = rank_categories.cat.codes

X = augdf.drop(columns=['Rank'])
y = augdf['Rank']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE는 훈련 데이터에만 적용 (data leakage 방지)
# k_neighbors=2: Silver처럼 샘플이 극히 적은 클래스에서 직선 보간 문제 방지
smote = SMOTE(k_neighbors=2, random_state=42)
X_train_aug, y_train_aug = smote.fit_resample(X_train, y_train)

print(f"---------------------------")
print(f"SMOTE 적용 전 훈련 데이터: {X_train.shape[0]}행")
print(f"SMOTE 적용 후 훈련 데이터: {X_train_aug.shape[0]}행")
print()

aug_plot_df = pd.DataFrame(X_train_aug, columns=X.columns)
aug_plot_df['Rank'] = [rank_map[r] for r in y_train_aug]
plot_scatter(aug_plot_df, "SMOTE Applied")

scaler = StandardScaler()
X_train_aug = scaler.fit_transform(X_train_aug)
X_test_aug = scaler.transform(X_test)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_aug, y_train_aug)

accuracy = svm_model.score(X_test_aug, y_test)
print(f"---------------------------")
print(f"데이터 증강 (SMOTE) + 표준화 :")
print(f"정확도 : {accuracy:.4f}")
results['SMOTE + StandardScaler'] = accuracy

sample = X_test_aug[0].reshape(1, -1)
prediction = svm_model.predict(sample)
print(f"예측된 랭크 : {rank_map[prediction[0]]}")
print()

## 이상치 제거 + 데이터 증강 (SMOTE)
# 전체 기준 IQR은 Silver(7개) 전체를 이상치로 판정해 삭제하므로
# 클래스별로 IQR을 적용해 각 랭크 내부 이상치만 제거
augdf2 = ori_data.drop(columns=['Player_Name', 'Unnamed: 0'])
augdf2 = augdf2.dropna()

numeric_cols2 = augdf2.select_dtypes(include=[np.number]).columns
clean_parts = []
for rank, group in augdf2.groupby('Rank'):
    Q1 = group[numeric_cols2].quantile(0.25)
    Q3 = group[numeric_cols2].quantile(0.75)
    IQR2 = Q3 - Q1
    mask2 = ~((group[numeric_cols2] < (Q1 - 1.5 * IQR2)) | (group[numeric_cols2] > (Q3 + 1.5 * IQR2))).any(axis=1)
    clean_parts.append(group[mask2])
augdf2 = pd.concat(clean_parts).copy()

print(f"클래스별 IQR 제거 후: {augdf2['Rank'].value_counts().to_dict()}")

rank_categories2 = augdf2['Rank'].astype('category')
rank_map2 = dict(enumerate(rank_categories2.cat.categories))
augdf2['Rank'] = rank_categories2.cat.codes

X2 = augdf2.drop(columns=['Rank'])
y2 = augdf2['Rank']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

smote2 = SMOTE(k_neighbors=2, random_state=42)
X_train_aug2, y_train_aug2 = smote2.fit_resample(X_train2, y_train2)

print(f"---------------------------")
print(f"[이상치 제거 후] SMOTE 적용 전: {X_train2.shape[0]}행")
print(f"[이상치 제거 후] SMOTE 적용 후: {X_train_aug2.shape[0]}행")
print()

aug_plot_df2 = pd.DataFrame(X_train_aug2, columns=X2.columns)
aug_plot_df2['Rank'] = [rank_map2[r] for r in y_train_aug2]
plot_scatter(aug_plot_df2, "Remove Outliner + SMOTE Applied")

scaler2 = StandardScaler()
X_train_aug2 = scaler2.fit_transform(X_train_aug2)
X_test_aug2 = scaler2.transform(X_test2)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_aug2, y_train_aug2)

accuracy = svm_model.score(X_test_aug2, y_test2)
print(f"---------------------------")
print(f"이상치 제거 + SMOTE + 표준화 :")
print(f"정확도 : {accuracy:.4f}")
results['Outliner X + SMOTE'] = accuracy

sample = X_test_aug2[0].reshape(1, -1)
prediction = svm_model.predict(sample)
print(f"예측된 랭크 : {rank_map2[prediction[0]]}")
print()

## 전처리 방식별 정확도 비교 시각화
plt.figure(figsize=(12, 5))
bars = plt.bar(results.keys(), results.values(), color=['gray', 'skyblue', 'steelblue', 'salmon', 'tomato', 'mediumseagreen', 'darkcyan'])
plt.ylim(0, 1.1)
plt.ylabel("Accuracy")
plt.title("Data Analysis + SVM Algorithm")
plt.xticks(rotation=15, ha='right')
for bar, val in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.4f}", ha='center', fontsize=10)
plt.tight_layout()
plt.show()