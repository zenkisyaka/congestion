import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta

# データ読み込み (適宜変更)
df = pd.read_csv("prohet/dummy_parking_data_with_weather.csv")  # 適宜ファイル名を変更してください
df["date_time"] = pd.to_datetime(df["date_time"])  # 日時を変換

# Prophet のカラム形式に変更
df.rename(columns={"date_time": "ds", "usage": "y"}, inplace=True)

# Prophet モデルの学習
model = Prophet()
model.fit(df)

# 予測期間の設定（1時間ごとのデータを未来7日間生成）
today = datetime.today().date()
future_dates = pd.date_range(start=today + timedelta(days=1), periods=7 * 24, freq="H")  # 1時間ごと
future_df = pd.DataFrame(future_dates, columns=["ds"])

# 予測
forecast = model.predict(future_df)

# 1日単位で `yhat` の平均を計算
forecast["date"] = forecast["ds"].dt.date  # 日付だけ抽出
forecast_daily = forecast.groupby("date")["yhat"].mean().reset_index()

# 結果を表示
print(forecast_daily)
