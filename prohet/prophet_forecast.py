import pandas as pd
import pickle
from datetime import datetime, timedelta

# 学習時のデータを読み込む
df_prophet = pd.read_csv("prohet/Updated_Parking_Data.csv")
df_prophet["date_time"] = pd.to_datetime(df_prophet["date_time"])  # 日時を変換

# One-Hot Encoding を適用
df_prophet = pd.get_dummies(df_prophet, columns=['weather', 'event'])

# Prophet のデータ形式に変更
df_prophet = df_prophet.rename(columns={"date_time": "ds", "usage": "y"})

# 学習済み Prophet モデルをロード
with open("prohet/prophet_model.pkl", "rb") as f:
    model = pickle.load(f)

# 未来の日付データを作成（警告対応: `freq="H"` → `freq="h"`）
today = datetime.today().date()
future_dates = pd.date_range(start=today + timedelta(days=1), periods=7 * 24, freq="h")  # 修正済み
future_df = pd.DataFrame(future_dates, columns=["ds"])

# 学習データ (`df_prophet`) を参照し、未来データの One-Hot 変数を埋める
for col in df_prophet.columns:
    if col not in ['ds', 'y']:  # 'ds'（日付）と 'y'（使用率）は除外
        future_df[col] = df_prophet[col].mean()  # 過去データの平均値を適用

# 予測を実行
forecast = model.predict(future_df)

# 時間帯を考慮した `yhat` の日ごとの平均を計算
forecast["hour"] = forecast["ds"].dt.hour  # 時間を抽出 8~22時
forecast_filtered = forecast[(forecast["hour"] >= 8) & (forecast["hour"] <= 22)].copy()
forecast_filtered["date"] = forecast_filtered["ds"].dt.date  # 日付だけ抽出
forecast_daily = forecast_filtered.groupby("date")["yhat"].mean().reset_index()

# 結果を表示
print(forecast_daily)