import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st


Home, Learn, Proses, Model, Implementasi = st.tabs(['Beranda', 'Tentang Dataset', 'Persiapan Data', 'Model', 'Implementasi'])

with Home:
   st.title("""Klasifikasi Kwalitas Air Menggunakan Metode K-Nearest Neighbor (KNN)
   """)
   st.write('Kelompok 2')
   st.text("""
            1. Dewi Imani Al Qur'Ani 200411100014
            2. Nuskhatul Haqqi 200411100034   
            """)


with Learn:
   st.title("""Klasifikasi Kwalitas Air Menggunakan Metode K-Nearest Neighbor (KNN)
   """)
   st.write('Pada Penelitian ini akan dilakukan pengklasifikasian kwalitas air menggunakan metode KNN. ')
   st.write('Dalam Klasifikasi ini data yang digunakan di ambil dari kaggle.')
   st.title('Klasifikasi data inputan berupa : ')
   st.write(""" 1. ph: pH 1. air (0 sampai 14). """)
   st.write(""" 2. Kekerasan: Kapasitas air untuk mengendapkan sabun dalam mg/L. """)
   st.write(""" 3. Padatan: Total padatan terlarut dalam ppm. """)
   st.write(""" 4. Chloramines: Jumlah Chloramines dalam ppm. """)
   st.write(""" 5. Sulfat: Jumlah Sulfat yang dilarutkan dalam mg/L. """)
   st.write(""" 6. Konduktivitas: Konduktivitas listrik air dalam μS/cm. """)
   st.write(""" 7. Karbon_organik: Jumlah karbon organik dalam ppm. """)
   st.write(""" 8. Trihalometana: Jumlah Trihalometana dalam μg/L. """)
   st.write(""" 9. Kekeruhan: Ukuran sifat air yang memancarkan cahaya di NTU. """)
   st.write(""" 10. Dapat Diminum: Menunjukkan apakah air aman untuk dikonsumsi manusia. Dapat Diminum = 1 dan Tidak Dapat Diminum = 0 """)
   st.title("""Asal Data""")
   st.write("Dataset yang digunakan adalah data kaggle pada link berikut https://www.kaggle.com/code/renzwicked/classification-water-quality/input")
   st.write("Total datanya yang digunakan ada 3276 data dengan 10 inputan terdiri dari 9 fitur dengan 1 label")
   data = pd.read_csv('water_potability.csv',encoding= 'unicode_escape')


with Proses:
   st.title("""Persiapan Data""")
   "### Dataset"
   data
   "### Pengecekan data yang kosong (Missing Value)"
   st.write(data.isnull().sum())
   """ 
   Melakukan pengecekan apakah didalam dataset terdapat data yang bernilai null atau tidak. Setelah dilakukan pengecekan
   dihasilkan bahwa pada dataset terdapat data yang bernilai null diantaranya 491 data dari fitur ph, 781 data dari fitur Sulfate, dan 181 data dari fitur Trihalomethanes.
   """
   """
   Karena dari pengecekan isnull terdapat 3 fitur yang memiliki nilai null maka dilakukan penanganan untuk nilai nullnya dengan mengisikan data yang bernilai null dengan 
   median atau nilai tengah dari data yang ada dalam setiap fitur.
   """
   data['Sulfate'] = data['Sulfate'].fillna(value=data['Sulfate'].median())
   data['ph'] = data['ph'].fillna(value=data['ph'].median())
   data['Trihalomethanes'] = data['Trihalomethanes'].fillna(value=data['Trihalomethanes'].median())
   "### Pengecekan data yang kosong (Missing Value) setelah dilakukan penanganan"
   st.write(data.isnull().sum())
   """
   Berikut dataset yang siap digunakan:
   """
   data         

with Model:
   st.title("""Modelling""")
   # memisahkan antara fitur dengan label
   X = data.iloc[:,0:9].values
   y = data.iloc[:,-1].values
   

   # melakukan split data sebesar 20% testing dan 80% training
   X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.fit_transform(X_test)

   # Menjalankan knn dengan k = 9
   knn_clf = KNeighborsClassifier(n_neighbors=15, metric = 'euclidean')
   knn_clf.fit(X_train,y_train)
   # Menjalankan prediksi
   y_pred = knn_clf.predict(X_test)

   akurasi = round(100 * accuracy_score(y_test,y_pred))
   st.subheader("Metode Yang Digunakan Adalah K-Nearest Neighbor (KNN)")
   st.write("Akurasi Terbaik Dari Skenario Uji Coba Diperoleh Sebesar : {0:0.2f} %" . format(akurasi))

   with open('knn_pickle','wb') as r:
      pickle.dump(knn_clf,r)
   with open('scaler_pickle','wb') as r:
      pickle.dump(scaler,r)


with Implementasi:
   st.title("""Implementasi Data""")
   col1,col2 = st.columns(2)
   with col1:
      ph = st.number_input('Masukkan PH air (0 sampai 14)')
   with col2:
      kekerasan = st.number_input('Masukkan Kekerasan dalam mg/L')
   with col1:
      padatan = st.number_input('Masukkan Padatan dalam ppm')
   with col2:
      Chloramines = st.number_input('Masukkan Chloramines dalam ppm')
   with col1:
      Sulfat = st.number_input('Masukkan Sulfat dalam mg/L')
   with col2:
      Konduktivitas = st.number_input('Masukkan Konduktivitas dalam μS/cm')
   with col1:
      Karbon_organik = st.number_input('Masukkan Karbon organik dalam ppm')
   with col2:
      Trihalometana = st.number_input('Masukkan Trihalometana dalam μg/L')
   with col1:
      Kekeruhan = st.number_input('Masukkan Kekeruhan')
   with col2:
      inputan = [ph,kekerasan,padatan,Chloramines,Sulfat,Konduktivitas,Karbon_organik,Trihalometana,Kekeruhan]



   def submit():
      # input
      inputs = np.array(inputan)
      inputs = inputs.reshape(1, -1)

      with open('knn_pickle', 'rb') as r:
         model_knn = pickle.load(r)
      with open('scaler_pickle', 'rb') as r:
         scaler_ = pickle.load(r)

      X_pred = model_knn.predict(scaler_.transform(inputs))
      if X_pred[0] == 1:
         h = 'Dapat Diminum'
      else:
         h= 'Tidak Dapat Diminum'
      hasil =f"Berdasarkan data yang Anda masukkan, maka ulasan masuk dalam kategori  : {h}"
      if (X_pred[0] == 'positif'):
         st.success(hasil)
      else:
         st.warning(hasil)

   all = st.button("Submit")
   if all :
      st.balloons()
      submit()







