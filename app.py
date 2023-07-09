import joblib
from datetime import datetime as dt
import streamlit as st
import numpy as np
from prophet import Prophet
import pandas as pd
from japanmap import pref_names, pref_code, picture
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.pyplot as plt

# デフォルト時刻の定義
now = dt.now()

# メインタイトル
st.title('東京電力管内　電力需要予測')


# 東京電力の電力供給エリアを日本地図で表示
st.subheader('電力予測エリア')
fig = plt.figure(figsize=(15, 15))

plt.imshow(picture({'東京':'blue', '神奈川':'blue', '埼玉':'blue', '千葉':'blue', '茨城':'blue',
                    '栃木':'blue', '群馬':'blue', '山梨':'blue', '静岡':'blue'}))
st.pyplot(fig)


# サイドバーに記載する予測条件の入力項目
side_menue = st.sidebar.header('予測日・予想気温情報入力')
select_data = st.sidebar.date_input('⚪️日付選択', now)
week = select_data.strftime('%A')
hours = list(range(0, 24))
select_time = st.sidebar.selectbox('⚪️時間選択', hours, index=hours.index(now.hour))


#　月毎にモデルに使用する気温情報の選択肢を変更するための表示画面設定
#　３月〜６月の表示
if (select_data.month >= 3) and (select_data.month <= 6):
    temprature = st.sidebar.slider('⚪️予想気温', 5, 35, 20, 1)
    # モデルの予測に使用するための数値変換
    if temprature >= 30:
        temp = 1
    else:
        temp = 0

 # 10、11月の表示
if  (select_data.month >= 10) and (select_data.month <= 11):
    temprature = st.sidebar.slider('⚪️予想気温', 5, 35, 20, 1)
    # モデルの予測に使用するための数値変換
    if temprature >= 30:
        temp = 1
    else:
        temp = 0

 # 7月〜9月の真夏日での表示
if (select_data.month >= 7) and (select_data.month <= 9):
    temprature = st.sidebar.slider('⚪️予想気温', 20, 40, 25, 1)
    # モデルの予測に使用するための数値変換
    if temprature == 30:
        temp = 0.2
    elif temprature == 31:
        temp = 0.4
    elif temprature == 32:
        temp = 0.6
    elif temprature == 33:
        temp = 0.8
    elif temprature >= 34:
        temp = 1
    else:
        temp = 0

 # 12月〜２月までの真冬日の表示
if (select_data.month >= 1) and (select_data.month <= 2) or (select_data.month == 12):
    temprature = st.sidebar.slider('⚪️予想気温', -10, 15, 5, 1)
    # モデルの予測に使用するための数値変換
    if 5 <= temprature  <= 6:
        temp =  0.2
    elif 3 <=temprature  <= 4:
        temp =  0.4
    elif 1 <= temprature  <= 2:
        temp =   0.6
    elif -1 <= temprature  <= 0:
        temp  =  0.8
    elif temprature <= -2:
        temp =  1
    else:
        temp =  0


# 曜日(week)の表記を英語から漢字に変更
if week == 'Monday':
    week ='月'
elif week == 'Tuesday':
    week ='火'
elif week == 'Wednesday':
    week = '水'
elif week == 'Thursday':
    week = '木'
elif week == 'Friday':
    week = '金'
elif week == 'Saturday':
    week = '土'
elif week == 'Sunday':
    week = '日'


 # 選択した日付、時刻情報を表示
st.sidebar.write(f"## 需要予測日:{select_data}({week}) {select_time}:00") 
st.sidebar.write(f"## 予想気温:{temprature}℃")
button = st.sidebar.button("実行")


# モデルのロード
if (select_data.month >= 3) and (select_data.month <= 6) or (select_data.month >= 10) and (select_data.month <= 11):
    model = joblib.load('model_norm_elec.pkl')
elif (select_data.month >= 7) and (select_data.month <= 9):
    model = joblib.load('model_hot_elec.pkl')
elif (select_data.month >= 1) and (select_data.month <= 2) or (select_data.month == 12):
    model = joblib.load('model_cold_elec.pkl')


# 予測に使用する日時・気温に関する変数を設定
pred_time = f"{select_data} {select_time}:00"
pred_date = f"{select_data}"
pred_temp = f"{temp}"


# 1日の需要予測をするために気温情報の24時間分のリストを作成
time = list(range(0, 24, 1))
day_pred = [f"{select_data} {t}:00:00" for t in time]
temp_zero = [0 for _ in range(24)]
temp_zero_two = [0.2 for _ in range(24)]
temp_zero_four = [0.4 for _ in range(24)]
temp_zero_six = [0.6 for _ in range(24)]
temp_zero_eight = [0.8 for _ in range(24)]
temp_one = [1 for _ in range(24)]


# 予測日のデータフレームを作成
 #　サイドバーで選択した１時間のみの需要予測用
future = pd.DataFrame([[pred_time, pred_temp]], columns=['ds','temp'])

 # サイドバーで選択した日付の需要予測用データフレームを選択した気温ごとに作成
if temp == 0:
    future_d = pd.DataFrame({'ds': day_pred, 'temp': temp_zero})
elif temp == 0.2:
    future_d = pd.DataFrame({'ds': day_pred, 'temp': temp_zero_two})
elif temp == 0.4:
    future_d = pd.DataFrame({'ds': day_pred, 'temp': temp_zero_four})
elif temp == 0.6:
    future_d = pd.DataFrame({'ds': day_pred, 'temp': temp_zero_six})
elif temp == 0.8:
    future_d = pd.DataFrame({'ds': day_pred, 'temp': temp_zero_eight})
elif temp == 1:
    future_d = pd.DataFrame({'ds': day_pred, 'temp': temp_one})
    

# アプリ起動時の名画面のデフォルト表示で、予測が完了した際に表示される情報を簡易表示
if button == False:
    st.write('左欄から条件を選択し、選択が終了したら『実行』ボタンを押してください。')
    st.markdown("## ____万kW")
    y_list = np.linspace(3000, 4000, 24)
    fig = px.bar(x=day_pred, y=y_list)
    fig.update_xaxes(title_text='時間')
    fig.update_yaxes(title_text='予測需要量(万kw)')
    st.plotly_chart(fig, use_container_width=True)


# 予測を実行
if button == True:
    forecast = model.predict(future)
    day_forecast = model.predict(future_d)
    # 予測結果を表示
    st.write(f"##### {select_data}({week}){select_time}:00 予想電力需要量")
    st.markdown(f"## {forecast['yhat'].iloc[0].astype(int)}万kW")
    desc_pred_time = pd.DataFrame({'ds':day_pred})
    desc_pred_time['ds'] = pd.to_datetime(desc_pred_time['ds'], format='%Y-%m-%d %H:%M')
    pred_df = pd.merge(desc_pred_time, day_forecast, on='ds')

    state_list = ['ds', 'yhat']

    drop_list = [i for i in pred_df if i not in state_list]
    pred_df.drop(columns=drop_list, inplace=True)     

    t = []
    for d in pred_df['ds']:
        timestamp = pd.Timestamp(d)
        hour = timestamp.hour
        time = f"{hour}:00"
        t.append(time)
    pred_df['ds'] = t
    pred_df = pred_df.rename(columns={'ds':'time', 'yhat':'predict'})
    
    # 1時間毎の予測結果の棒グラフを作成
    fig = px.bar(pred_df, x=day_pred, y='predict')
    fig.update_xaxes(title_text='時間')
    fig.update_yaxes(title_text='予測需要量(万kw)')
    st.plotly_chart(fig, use_container_width=True)