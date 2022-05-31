
import streamlit as st
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pydeck as pdk
import geopy.distance
scaler = StandardScaler()

header = st.container()

with header:
    st.title("Upload your csv file here")
    st.text("make sure it has an X and Y fields with capital letters")
    data_file = st.file_uploader("Upload CSV", type=['csv'])
    if data_file is not None:
        full_data = pd.read_csv(data_file)
        time = full_data.tracktime
        time2 = time[:].str.split()
        time3 = [x[1] for x in time2]
        time5 = [x.split(":") for x in time3]
        time6 = [x[0:2] for x in time5]
        time61 = [float(x[0]) * 60 for x in time6]
        time62 = [int(x[1]) for x in time6]
        time7 = np.add(time61, time62)
        x_ch = abs(full_data.X.diff()) < 0.01
        x_ch1 = full_data.X.copy()
        x_ch2 = np.diff(x_ch1, prepend=[full_data.X[0]])
        x_ch2[~x_ch] = 0
        y_ch = abs(full_data.Y.diff()) < 0.1
        y_ch1 = full_data.Y.copy()
        y_ch2 = np.diff(y_ch1, prepend=[full_data.Y[0]])
        y_ch2[~y_ch] = 0
        dist = np.sqrt((y_ch2 ** 2) + (x_ch2 ** 2))
        diff_dis = np.diff(dist, prepend=[dist[0]])

        new_data = full_data.copy()
        new_data["first_gradiant_mag"] = dist
        new_data["secand_gradiant_mag"] = diff_dis
        new_data["angel"] = 180. * np.arctan2(y_ch2, x_ch2) / np.pi
        # new_data["angel_diff"]=np.diff(new_data["angel"],prepend=[new_data["angel"][0]])
        Y = new_data["track"]
        mag_mean = new_data[['X','Y']].copy()
        mag_mean["mean"] = float(0)
        for idx_row in range(mag_mean.shape[0] - 5):
            for idx5 in range(1, 6):
                loc_dist = np.sqrt(((mag_mean["X"].values[idx_row + idx5] - mag_mean["X"].values[idx_row]) ** 2) + (
                            (mag_mean["Y"].values[idx_row + idx5] - mag_mean["Y"].values[idx_row]) ** 2))
                if loc_dist < 0.1:
                    mag_mean["mean"].values[idx_row] += loc_dist
            mag_mean["mean"].values[idx_row] = mag_mean["mean"].values[idx_row] / 5
        new_data["mean"] = mag_mean["mean"]
        mag_std = new_data[['X','Y']].copy()
        mag_std["std"] = float(0)
        for idx_row in range(mag_mean.shape[0] - 5):
            diff_std = []
            for idx5 in range(1, 6):
                loc_dist = np.sqrt(((mag_std["X"].values[idx_row + idx5] - mag_std["X"].values[idx_row]) ** 2) + (
                            (mag_std["Y"].values[idx_row + idx5] - mag_std["Y"].values[idx_row]) ** 2))
                if loc_dist < 0.1:
                    diff_std.append(loc_dist)
                mag_std["std"].values[idx_row] = np.std(diff_std)
        new_data["std"] = mag_std["std"]
        new_data.fillna(0, inplace=True)
        X = new_data[['first_gradiant_mag','secand_gradiant_mag','angel','mean','std']].copy()
        X = scaler.fit_transform(X)

        with open('best_lin.pkl', 'rb') as f:
            best_rfc = pickle.load(f)
        rfc_pred_y = best_rfc.predict(X)


        # def plt_2d(X_pca, y, title):
        #     fig = plt.figure(figsize=(16, 16))
        #     ax = fig.add_subplot(111, aspect='equal')
        #     ax.scatter(X_pca['X'][y == 1], X_pca['Y'][y == 1], color='r')
        #     ax.scatter(X_pca['X'][y == 0], X_pca['Y'][y == 0], color='b')
        #     ax.legend(('Moving', 'standing'))
        #     ax.set_xlabel('$X$')
        #     ax.set_ylabel('$Y$')
        #     ax.set_title(title)
        #     ax.grid(True)
        #     st.pyplot(fig)
        # plt_2d(new_data, rfc_pred_y, "prodicted data")
        map_frame = pd.DataFrame()
        new_full_data = full_data.copy()
        new_full_data["predicted_state"] = rfc_pred_y
        flag = 0
        count = 0
        for idx, value in enumerate(new_full_data["predicted_state"]):
            if value != flag:
                if flag == 0:
                    flag = 1
                    count = idx
                else:
                    flag = 0
                    timer = int(time7[idx]) - int(time7[count])
                    if timer <= 0:
                        timer = timer + 720
                    distanc = geopy.distance.geodesic([new_full_data["X"][idx], new_full_data["Y"][idx]],
                                                      [new_full_data["X"][count], new_full_data["Y"][count]]).km
                    speed = float(distanc / (timer / 60))
                    if speed > 5:
                        for i in range(count, idx):
                            new_full_data["predicted_state"][i] = 2

        map_frame["lon"] = new_data['X']
        map_frame["lat"] = new_data['Y']
        map_frame["color"] = new_full_data["predicted_state"]
        # st.map(map_frame)

        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=map_frame["lat"][0],
                longitude=map_frame["lon"][0],
                 zoom=9,
                # pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=map_frame[map_frame["color"]==1],
                    get_position='[lon, lat]',
                    get_color='[255,94,87]',
                    get_radius=200,
                    radiusMinPixels=5,
                ),
                pdk.Layer(
                    'ScatterplotLayer',
                    data=map_frame[map_frame["color"] == 0],
                    get_position='[lon, lat]',
                    get_color='[75,205,250]',
                    get_radius=200,
                    radiusMinPixels=5,
                ),
                pdk.Layer(
                    'ScatterplotLayer',
                    data=map_frame[map_frame["color"] == 2],
                    get_position='[lon, lat]',
                    get_color='[125,125,125]',
                    get_radius=20,
                    radiusMinPixels=5,
                ),
            ],
        ))


        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')



        csv = convert_df(new_full_data)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='predicted_GPS.csv',
            mime='text/csv',
        )
        st.text("for support please contact asi640@gmail.com ")

