import urllib.request  
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error as MSE
import numpy as np

# ==========================================
# 1. Webスクレイピング & CSV保存
# ==========================================
url = 'https://baseball-data.com/24/stats/hitter2-all/tpa-1.html'
try:
    html = urllib.request.urlopen(url)
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find_all('table')[0]
    rows = table.find_all('tr')
except Exception as e:
    print(f"URLへのアクセス、またはテーブルの取得に失敗しました: {e}")
    exit()

headData = []
playerData = []

for i, row in enumerate(rows):
    if i == 0:
        head = row.find_all('th')
        for j in range(len(head)):
            headData.append(head[j].get_text())
    elif (i > 0) and (i + 1 < len(rows)):
        playerRow = []
        for playerValue in row.find_all('td'):
            playerRow.append(playerValue.get_text())
        playerData.append(playerRow)

df = pd.DataFrame(playerData, columns=headData)
df.to_csv('playerData.csv', sep=',', header=True, index=False)


# ==========================================
# 2. データ読み込み & 型変換
# ==========================================
df = pd.read_csv('playerData.csv', sep=',')

# 選手名、チーム、守備などの文字列列以外を、一括で数値型に変換
# 変換できない文字列（'-' など）は NaN（欠損値）になり、その後 0 に置換
cols_to_numeric = df.columns.difference(['選手名', 'チーム', '守備'])
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', ''), errors='coerce')
df = df.fillna(0) 

# ==========================================
# 3. データの分割・分析
# ==========================================
df11d = df[df['チーム'] != '中日'].copy()
df11g = df[df['チーム'] != '巨人'].copy()
df_d = df[df['チーム'] == '中日'].copy()
df_g = df[df['チーム'] == '巨人'].copy()

numeric_cols_d = df11d.select_dtypes(include=[np.number])
numeric_cols_g = df11g.select_dtypes(include=[np.number])

df_daritud = pd.DataFrame(numeric_cols_d.corr()['打率'])
df_homerund = pd.DataFrame(numeric_cols_d.corr()['本塁打'])
df_daritug = pd.DataFrame(numeric_cols_g.corr()['打率'])
df_homerung = pd.DataFrame(numeric_cols_g.corr()['本塁打'])

X_daritud_columns = df_daritud[df_daritud['打率'] >= 0].index.tolist()
X_homerund_columns = df_homerund[df_homerund['本塁打'] >= 0.6].index.tolist()
X_daritug_columns = df_daritug[df_daritug['打率'] >= 0].index.tolist()
X_homerung_columns = df_homerung[df_homerung['本塁打'] >= 0.6].index.tolist()

if '打率' in X_daritud_columns: X_daritud_columns.remove('打率')
if '本塁打' in X_homerund_columns: X_homerund_columns.remove('本塁打')
if '打率' in X_daritug_columns: X_daritug_columns.remove('打率')
if '本塁打' in X_homerung_columns: X_homerung_columns.remove('本塁打')

# ==========================================
# 4. 機械学習モデルの学習と予測
# ==========================================
# 中日用モデル
X_daritud = df11d[X_daritud_columns]
y_daritud = df11d['打率']
X_homerund = df11d[X_homerund_columns]
y_homerund = df11d['本塁打']

# 巨人用モデル
X_daritug = df11g[X_daritug_columns]
y_daritug = df11g['打率']
X_homerung = df11g[X_homerung_columns]
y_homerung = df11g['本塁打']

# 分割
X_train_daritud, X_test_daritud, y_train_daritud, y_test_daritud = train_test_split(X_daritud, y_daritud, random_state=1)
X_train_homerund, X_test_homerund, y_train_homerund, y_test_homerund = train_test_split(X_homerund, y_homerund, random_state=1)
X_train_daritug, X_test_daritug, y_train_daritug, y_test_daritug = train_test_split(X_daritug, y_daritug, random_state=1)
X_train_homerung, X_test_homerung, y_train_homerung, y_test_homerung = train_test_split(X_homerung, y_homerung, random_state=1)

# 学習
lr_daritud, lr_homerund = LR(), LR()
lr_daritud.fit(X_train_daritud, y_train_daritud)
lr_homerund.fit(X_train_homerund, y_train_homerund)

lr_daritug, lr_homerung = LR(), LR()
lr_daritug.fit(X_train_daritug, y_train_daritug)
lr_homerung.fit(X_train_homerung, y_train_homerung)

# 評価の出力（RMSE）
print(f"中日モデル RMSE - 打率: {np.sqrt(MSE(y_test_daritud, lr_daritud.predict(X_test_daritud))):.4f}, 本塁打: {np.sqrt(MSE(y_test_homerund, lr_homerund.predict(X_test_homerund))):.4f}")
print(f"巨人モデル RMSE - 打率: {np.sqrt(MSE(y_test_daritug, lr_daritug.predict(X_test_daritug))):.4f}, 本塁打: {np.sqrt(MSE(y_test_homerung, lr_homerung.predict(X_test_homerung))):.4f}\n")

# 本番予測
y_pred_d_daritu = np.clip(lr_daritud.predict(df_d[X_daritud_columns]), 0, None)
y_pred_d_homerun = np.clip(lr_homerund.predict(df_d[X_homerund_columns]), 0, None)
y_pred_g_daritu = np.clip(lr_daritug.predict(df_g[X_daritug_columns]), 0, None)
y_pred_g_homerun = np.clip(lr_homerung.predict(df_g[X_homerung_columns]), 0, None)

df_d['予測本塁打'] = np.round(y_pred_d_homerun).astype(np.int32)
df_d['予測打率'] = np.round(y_pred_d_daritu, 3)
df_g['予測本塁打'] = np.round(y_pred_g_homerun).astype(np.int32)
df_g['予測打率'] = np.round(y_pred_g_daritu, 3)

# ==========================================
# 5. 結果表示
# ==========================================
df_d_sorted_desc = df_d.sort_values(by=['本塁打', '打率'], ascending=[False, False])
print("--- 中日ドラゴンズ 予測結果 (本塁打・打率 降順) ---")
print(df_d_sorted_desc[['選手名', '打率', '予測打率', '本塁打', '予測本塁打']].to_string(index=False))

print("\n")

df_g_sorted_desc = df_g.sort_values(by=['本塁打', '打率'], ascending=[False, False])
print("--- 読売ジャイアンツ 予測結果 (本塁打・打率 降順) ---")
print(df_g_sorted_desc[['選手名', '打率', '予測打率', '本塁打', '予測本塁打']].to_string(index=False))