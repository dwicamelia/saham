#library streamlit
import streamlit as st
from pyrebase import pyrebase
import yfinance as yf
import pandas as pd
import cufflinks as cf
#library tanggal
from datetime import timedelta, date
import datetime 
import time
from datetime import time
from math import sqrt
import math
import numpy as np
#library normalisasi dan regresi linear
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error
#library plot grafik #tidak bisa ditampilkan di streamlit
import matplotlib.pyplot as plt
import plotly
from matplotlib import style
from sklearn import tree
import plotly.graph_objects as go

#path
#Config Key
firebaseConfig = {
    'apiKey': "AIzaSyDsQVkW_4JHQDZxFrYuj16We8pjFd9hHYI",
    'authDomain': "prediksi-saham-dwi.firebaseapp.com",
    'projectId': "prediksi-saham-dwi",
    'databaseURL':"https://prediksi-saham-dwi-default-rtdb.asia-southeast1.firebasedatabase.app",
    'storageBucket': "prediksi-saham-dwi.appspot.com",
    'messagingSenderId': "317059044357",
    'appId': "1:317059044357:web:112de7368d46f990abb86c",
    'measurementId': "G-43V5KJVHC4"
  }
#firebase auth
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
storage = firebase.storage()

#Database
db = firebase.database()


st.sidebar.title("Prediksi saham App")

#Auth
choice = st.sidebar.selectbox('login/registrasi', ['Login','Registrasi'])
email = st.sidebar.text_input('masukkan alamat email anda:')
password = st.sidebar.text_input('Masukkan password anda:', type= 'password')




if choice == 'Registrasi':
        nama = st.sidebar.text_input('Please input name', value='Default')
        regis = st.sidebar.button('Buat akun')

        if regis:
            user = auth.create_user_with_email_and_password(email, password)
            st.success('Akun sudah berhasil dibuat')
            st.balloons()

            #Regis
            user = auth.sign_in_with_email_and_password(email, password)
            db.child(user['localId']).child("Nama").set(nama)
            db.child(user['localId']).child("ID").set(user['localId'])
            st.info("Login via login dropdown selection")
            st.title("Wellcome\n" +nama)
            st.info("Login via login dropdown selection")
        else:
            st.error('Periksa kembali email/password anda')
       

    

if choice == 'Login':
        # Login button
        login = st.sidebar.checkbox('Login')
        if login:
            #Login
            user = auth.sign_in_with_email_and_password(email, password)
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            st.balloons()
            st.info("Anda berhasil login")
            st.info("silahkan pilih menu")
            st.title("Menu Proses")
            pilih = st.radio('Jump to Menu', ['prediksi','Proses file','Komparasi Trend grafik harga'])
        else:      
            st.error('Harus Login dulu/Registrasi')
            st.write("Sorry, Kamu tidak memiliki akses untuk ke Menu fitur!")
        


    ##Menu Proses Fitur####

if pilih == 'Komparasi Trend grafik harga':
                    st.title("Komparasi grafik Harga Saham")
                    st.subheader("Berikut adalah list saham: LQ45")
                    st.write("Pilih Saham sesuai pilihan anda")
                    # start_date = st.date_input("Tanggal Mulai", datetime.date(2020,7,22))
                    # end_date = st.date_input("Tanggal Akhir", datetime.datetime.now())
                    # today = datetime.datetime.now()
                    st.title("------------------------------------------------")
                    stocks = ("ADRO.JK", "AMRT.JK", "ANTM.JK", "ASII.JK", "BBCA.JK", "BBNI.JK", 
                        "BBRI.JK","BBTN.JK","BFIN.JK","BMRI.JK","BRPT.JK","BUKA.JK","CPIN.JK","EMTK.JK","ERAA.JK","EXCL.JK","GGRM.JK", "HMSP.JK",
                        "HRUM.JK", "ICBP.JK","INCO.JK","INDF.JK","INKP.JK","INTP.JK","ITMG.JK","JPFA.JK","KLBF.JK","MDKA.JK","MEDC.JK","MIKA.JK",
                        "MNCN.JK","PGAS.JK","PTBA.JK","PTPP.JK","SMGR.JK","TBIG.JK","TINS.JK","TKIM.JK","TLKM.JK","TOWR.JK","TPIA.JK","UNTR.JK",
                        "UNVR.JK","WIKA.JK","WSKT.JK")
                    selected_stocks= st.multiselect("Pilih Saham yang ingin di komparasi", stocks)
                    start = st.date_input('Start', value = pd.to_datetime('2019-6-14'))
                    end = st.date_input('End', value = pd.to_datetime('today'))
                        
                    if len(selected_stocks)>0:
                        df=yf.download(selected_stocks, start, end)['Close']
                        st.line_chart(df)


            


    # mengambil data saham
        #Upload_File

if pilih == 'Proses file':
            st.subheader("saham")
            file_saham = st.file_uploader("saham file", type=["csv"])
            if file_saham is not None:
                df = pd.read_csv(file_saham)
                # df.seek(0)
                df["Date"] = df["Date"].str.replace('/', '')
                #mengisi data <NA>
                df.fillna(value =-99999, inplace = True)
                df.dropna(inplace=True) # drop/hilangkan nilai yang "not a number"/NaN
                st.write(df)
                # file_container.write( df)
                columnsTiltles = ["Date","Open","High","Low","Volume","Close"]
                df = df.reindex(columns=columnsTiltles)
                X = df.iloc[:, 1:-1].values
                y = df.iloc[:, 5].values
                #mengisi data <NA>
                df.fillna(value =-99999, inplace = True)
                df.dropna(inplace=True) # drop/hilangkan nilai yang "not a number"/NaN

            else:
                st.info(
                f"""
                    👆 Upload a .csv file first. Sample to try: [biostats.csv](https://people.sc.fsu.edu/~jburkardt/data/csv/biostats.csv)
                    """
                    )

                st.stop()
                #buatprediksi
            if st.button('Prediksi'):
                    #Hasil predikisi
                st.write("Output Prediksi: ")
                jml_OutputPrediksi = int(7) # 1 persen / 100 = 0.01
                df['HargaPrediksi'] = df['Close'].shift(-jml_OutputPrediksi)
                #Linear Regresi
                df['Date'] = "2019-04-01".replace("-", "")
                X = np.array(df.drop(['HargaPrediksi'],1)) # X tidak termasuk kolom "output_prediksi"
                X = preprocessing.scale(X) # normalisasi nilai menjadi -1 hingga 1
                X_prediksi = X[-jml_OutputPrediksi:] # data X untuk prediksi (1 persen elemen terakhir)
                X = X[:-jml_OutputPrediksi] # data X (99 persen elemen)
                df.dropna(inplace=True) # drop/hilangkan nilai yang "not a number"/NaN
                y = np.array(df['HargaPrediksi']) # y adalah nilai output prediksi

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

                # Feature Scaling
                from sklearn.preprocessing import StandardScaler
                sc_X = StandardScaler()
                X_train = sc_X.fit_transform(X_train)
                X_test = sc_X.transform(X_test)
                sc_y = StandardScaler()


                # Fitting Multiple Linear Regression to the Training set
                from sklearn.linear_model import LinearRegression
                regressor = LinearRegression()
                regressor.fit(X_train, y_train)

                # Predicting the Test set results
                y_pred = regressor.predict(X_test)



                from sklearn.linear_model import LinearRegression
                regressorr = LinearRegression()
                regressorr.fit(X_train, y_train)



                st.write('Hasil Prediksi Harga')
                setPrediksi = regressorr.predict(X_prediksi) # prediksi nilai y (output)
                st.write(setPrediksi)
                st.line_chart(setPrediksi)

                st.write("R-square")
                r2 = regressorr.score(X_test, y_test)
                st.write("The Coefficient is: ", r2)

                st.write("Mean Square error")
                mse=mean_squared_error(y_test, y_pred)
                st.write(mse)

                st.write("Root Mean Square error")

                rmse=mean_squared_log_error(y_test, y_pred) 
                st.write(rmse)

                st.write("Mean Absolute Percentage error")
                mape = mean_absolute_percentage_error(y_test, y_pred)
                st.write(mape)

                ####grafik
                df['HargaPrediksi'] = np.nan
                lastDate = df.iloc[-1].name # dapatkan tanggal terakhir
                lastSecond = lastDate
                oneDay = 86400 # detik =  1 hari
                nextSecond = lastSecond + oneDay

        #Legenda
                for i in setPrediksi : # untuk semua nilai yang telah di prediksi
                    nextDate = datetime.datetime.fromtimestamp(nextSecond) # hitung tanggal selanjutnya
                    nextSecond += 86400 # tambahkan detik selanjutnya menjadi satu hari berikutnya
                    df.loc[nextDate] = [np.nan for _ in range(len(df.columns)-1)]+[i] # tambahkan elemen i (nilai prediksi) 

                ##
                df['Close'].plot()
                df['HargaPrediksi'].plot()

                st.set_option('deprecation.showPyplotGlobalUse', False)
                tanggal = df['Date']
                plt.suptitle("Saham")
                plt.xlabel('Date')
                plt.ylabel("Harga")
                plt.legend()	    
                st.pyplot()
                plt.show()  
            
            # ###Dropfile
            # 
            
        #Batas_olahData


    ###Prediksi harga
if pilih == "prediksi":
        ### Read csv
        st.header("Daftar Nama dan Kode Saham Perusahaan")
        kd_saham = pd.read_excel('list_saham.xlsx')
        st.write(kd_saham)


        st.title("Dashboard Menu Prediksi")
        st.markdown("ini form input data")

        #Download Data from yahoofinance
        start_date = st.date_input("Tanggal Mulai", datetime.date(2020, 8, 8))
        end_date = st.date_input("Tanggal Akhir", datetime.datetime.now())
        today = datetime.datetime.now()

        stocks = st.text_input("Input Kode Saham : CONTOH : BMRI.JK (Bank Mandiri)", 'BMRI.JK')
        menu = ["Dataset","Model Linear Regresion"]
        pilih = st.selectbox("Model Prediksi", menu)
        Data = yf.download(stocks, country='indonesia', start = start_date , end = end_date)
        

        df = pd.DataFrame(data=Data)
        df.to_csv(''+stocks+'.csv')
        if(df.empty):
                #Format df
                # Last 2 yrs rows => 502, in ascending order => ::-1
                Data=data.head(503).iloc[::-1]
                Data=data.reset_index()
                #Keep Required cols only
                df=pd.DataFrame()
                df['Date']=data['hari']
                df['Open']=data['1. open']
                df['High']=data['2. high']
                df['Low']=data['3. low']
                df['Close']=data['4. close']
                df['Adj Close']=data['5. adjusted close']
                df['Volume']=data['6. volume']
                df.to_csv(''+stocks+'.csv',index=False)
                st.write(df)
            # return

        if pilih == "Dataset":
            st.write(df)
            st.line_chart(df['Close'])

            
        elif pilih == "Model Linear Regresion":
            #menentukan banyaknya output yang akan terprediksi
            st.write("Grafik Linear Regression: ")
            jml_HasilPrediksi = int(7) # 7 hari prediksi
            # menggeser tabel kolom "Close" ke atas sehingga beberapa nilai terakhir menjadi NaN
            Data['HasilPrediksi'] = Data['Close'].shift(-jml_HasilPrediksi)
        
            
            # #new data dengan data relevan
            # data_new = Data[['Close', 'HasilPrediksi']]

            #Struktur Data untuk Train, Test & Prediksi
            # X tidak termasuk kolom "Hasil prediksi"
            X = np.array(Data.drop(['HasilPrediksi'],1))
            # normalisasi nilai menjadi -1 hingga 1
            X = preprocessing.scale(X) 
            # data X untuk prediksi (1 persen elemen terakhir)
            X_to_be_prediksi = X[-jml_HasilPrediksi:]
            # data X (99 persen elemen)
            X = X[:-jml_HasilPrediksi]
            # drop/hilangkan nilai yang "not a number"/NaN 
            Data.dropna(inplace=True)
            # Y adalah nilai output prediksi
            Y = np.array(Data['HasilPrediksi'])

            #validation dengan ukuran data yang diuji 20% dan dilatih 80%
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)

            #regresi
            clf = LinearRegression(n_jobs=-1)
            #model Pelatihan
            clf.fit(X_train, y_train)

            #Model Pengujian
            y_test_pred=clf.predict(X_test)

            #Forecasting y
            forecast_set = clf.predict(X_to_be_prediksi)
            
            import matplotlib.pyplot as plt2
            fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
            plt2.suptitle("Linear Regression")
            plt2.plot(y_test,label='Actual Price' )
            plt2.plot(y_test_pred,label='Predicted Price')
            
            plt2.legend(loc=4)
            # plt2.savefig('LR.png')
            plt2.close(fig)
            st.pyplot(fig)
            rmse=mean_squared_log_error(y_test, y_test_pred)
            MAPE = mean_absolute_percentage_error(y_test, y_test_pred)
            st.write( "Mean Absolute Percentage Error:" ,MAPE)
            MSE = mean_squared_error(y_test, y_test_pred)
            st.write( "Mean squared Error:" ,MSE)
            st.write("Linear Regression RMSE:",rmse)
            r2 = clf.score(X_test, y_test)
            st.write("The Coefficient is: ",r2)

            
            # second_column.text("Data['HasilPrediksi']=np.nan")


            #Forecasting y
            forecast_set = clf.predict(X_to_be_prediksi)
            qf = cf.QuantFig(Data, legend='top',  name=stocks)  
            fig1 = qf.iplot(asFigure=True, dimensions=(800, 600), fill=True)
            

            mean=forecast_set.mean()
            lr_pred = forecast_set[0]
            st.write("##############################################################################")
            st.write("Hari Esok ",stocks," di Prediksikan Close Harga by Linear Regression: ",lr_pred)
            st.write("Hasil prediksi",forecast_set)
            #chart
            # import mplfinance as mpf
            # # Data = Data.set_index('Date')
            # mpf.plot(Data)
            
            st.line_chart(forecast_set)
            # st.write("Tanggal",prediksi)
            st.write("Hasil Rata-Rata Prediksi",mean)
            st.write("##############################################################################")
            st.write("Data Saham:",Data)

            

            # #df_grafik
            Data['HasilPrediksi']=np.nan
            lastDate = Data.iloc[-1].name # dapatkan tanggal terakhir
            lastSecond = lastDate.timestamp()
            oneDay = 86400 # detik =  1 hari
            nextSecond = lastSecond + oneDay
            for i in forecast_set : # untuk semua nilai yang telah di prediksi
                    nextDate = datetime.datetime.fromtimestamp(nextSecond) # hitung tanggal selanjutnya
                    nextSecond += 86400 # tambahkan detik selanjutnya menjadi satu hari berikutnya
                    Data.loc[nextDate] = [np.nan for _ in range(len(Data.columns)-1)]+[i] # tambahkan elemen i (nilai prediksi)

        #grafik prediksi
            st.title('Trend Grafik Harga saham\n' + stocks)
            Data['Close'].plot()
            Data['HasilPrediksi'].plot()
            plt.suptitle("Saham", name=stocks)
            plt.xlabel("Date")
            plt.ylabel("Harga Rp.")
            plt.legend()	    
            st.pyplot(plt)
            plt.show()
            plt.savefig(''+stocks+'.png') 

            st.set_option('deprecation.showPyplotGlobalUse', False)
            qf = cf.QuantFig(Data, legend='top',  name=stocks)  
            fig1 = qf.iplot(asFigure=True, dimensions=(800, 600), fill=True)
            # Render plot using plotly_chart
            st.plotly_chart(fig1)

            # convert df to json in db
            # df = df.to_json()
            user = auth.sign_in_with_email_and_password(email, password)
            data = stocks
            #csv
            csv = (''+stocks+'.csv')
            saham = csv
            #picture grafik
            pict = (''+stocks+'.png') 
            # convert Prediksi to json in db
            forecast_setInString = str(forecast_set)  # float -> str
            print(type(forecast_setInString))  # str
            prediksi= forecast_setInString
            db.child(user['localId']).child("Data").set(data)
            db.child(user['localId']).child("Datasaham").set(saham)
            db.child(user['localId']).child("Prediksi").set(prediksi)
            db.child(user['localId']).child("Grafik").set(pict)
            #SaveFile Database
            storage.child(''+stocks+'.png').put(pict)
            storage.child(''+stocks+'.csv').put(saham)
