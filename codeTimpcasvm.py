from tkinter import *
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
from turtle import bgcolor
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
#import PySimpleGUI as sg

df = pd.read_csv(r'C:\Users\trant\Desktop\tri tue nhan tao\DuDoanBenhTim\HeartAttackDataSet.csv')
# lấy ra các cột dữ liệu datatype  =  object
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
# lấy ra các cột dữ liệu datatype != object
num_cols = [col for col in df.columns if df[col].dtype != 'object']

# In ra các cột dữ liệu  datatype  =  object
for col in cat_cols:
    print(f"{col} has {df[col].unique()} values\n")
# Kiểm tra và sắp xếp các giá trị bị thiếu
df.isna().sum().sort_values(ascending = False)
# Kiểm tra các giá trị thiếu trong các cột datatype != object
df[num_cols].isnull().sum()
# Kiểm tra các giá trị thiếu trong các cột datatype = object
df[cat_cols].isnull().sum()
# điền các giá trị null, chúng ta sẽ sử dụng hai phương pháp:
# lấy mẫu ngẫu nhiên cho các giá trị null cao hơn 
# lấy mẫu trung bình/chế độ cho giá trị null thấp hơn

print("data info:")
print(df.info())

def random_value_imputation(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample
    
def impute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)

# Điền các giá trị null bằng phương pháp ngẫu nhiên random_value_imputation
for col in num_cols:
    random_value_imputation(col)

print("Điền các giá trị null bằng phương pháp ngẫu nhiên")
print(df[num_cols].isnull().sum())


# điền các giá trị còn thiếu của các cột datatype = object bằng pp impute_mode
for col in cat_cols:
    impute_mode(col)

# In ra số nhãn của các cột datatype = object
for col in cat_cols:
    print(f"{col} has {df[col].nunique()} categories\n")

# Xác định các giá trị ngoại lệ
def detect_outliers(data):
    outliers = []
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    for value in data:
        z_score = (value - mean) / std
        if np.abs(z_score) > threshold:
            outliers.append(value)
    return outliers

# Xử lý các giá trị ngoại lệ bằng cách thay thế chúng bằng giá trị trung bình
def handle_outliers(data):
    outliers = detect_outliers(data)
    if outliers:
        mean = np.mean(data)
        data = [mean if value in outliers else value for value in data]
    return data

# Áp dụng hàm handle_outliers cho mỗi cột dữ liệu
for col in num_cols:
    df[col] = handle_outliers(df[col])

print("Dữ liệu chưa mã hoá")
print(df.head())

le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# in ra dữ liệu đã mã hóa 
print("Dữ liệu đã mã hoá")
print(df.head())

X = np.array(df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']].values)    
y = np.array(df['target'])

# Tạo các tập X_train_main, X_test_main, y_train_main, y_test_main (70% train, 30% test)
X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(X, y, test_size=0.3 , shuffle = True)

# Tạo hàm sử dụng PCA
def PCA_method(formula):
    max = 0
    # Lặp qua các cột dữ liệu
    for j in range(1,14):
        print("Lần lặp", j)
        # Khởi tạo và áp dụng PCA
        pca = PCA(n_components = j)
        # Huấn luyện mô hình với tập huấn luyện.
        pca.fit(X)
        Xbar = pca.transform(X)
        # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(Xbar, y, test_size=0.3 , shuffle = True)
        # Kiểm tra công thức phân loại và huấn luyện mô hình
        # Nếu tham số đầu vào là svm thì thực hiện câu lệnh if
        if(formula == 'svm'):
            # Khởi tạo mô hình cây quyết định svm với tiêu chí entropy.            
            svmModel = svm.SVC(kernel='linear')
            # Huấn luyện mô hình với tập huấn luyện.
            svmModel.fit(X_train, y_train)
             #  Dự đoán nhãn cho tập kiểm tra.
            y_predict_svm = svmModel.predict(X_test)
             # Tính tỷ lệ dự đoán đúng của mô hình.
            rate = accuracy_score(y_test, y_predict_svm)
            print("Ty le du doan dung svm: ", rate)
            # Nếu tỷ lệ dự đoán đúng của mô hình hiện tại (rate) lớn hơn giá trị max hiện tại, cập nhật max, num_pca, pca_best, và modeImax với các giá trị tương ứng của mô hình tốt nhất.
            if(rate > max):
                num_pca = j
                pca_best = pca
                max = rate
                modeImax = svmModel
    # Trả về mô hình và cấu hình tốt nhất
    return modeImax, pca_best, num_pca

#chưa dùng PCA
#svm
svmModel = svm.SVC(kernel='linear')
svmModel.fit(X_train_main, y_train_main)

# Dùng PCA:
#svm
svm_PCA,pca_best_svm,num_pca_svm = PCA_method('svm')



# FORM
form = tk.Tk()
form.title("Dự đoán khả năng bị bệnh tim của bệnh nhân:")
form.geometry("1700x900")

lable_people = LabelFrame(form, text = "Nhập thông tin bệnh nhân", font=("Arial Bold", 13), fg="red")
lable_people.pack(fill="both", expand="yes")
lable_people.config(bg="#79b484")
# THÔNG TIN CỘT 1
lable_age = Label(form,font=("Arial Bold", 10), text = "Tuổi:" ,bg="#79b484").place(x = 180 , y = 50)
textbox_age = Entry(form,width = 30,font=("Arial Bold", 10))
textbox_age.place(x = 410 , y = 50)

lable_sex = Label(form,font=("Arial Bold", 10), text = "Giới tính:" ,bg="#79b484").place(x = 180 , y = 90)
lable_sex_gioitinh = ['Nam',  'Nữ']
lable_sex = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_sex_gioitinh, state = "readonly")
lable_sex.place(x = 410 , y = 90)
lable_sex.current(0)

lable_cp = Label(form,font=("Arial Bold", 10), text = "Loại đau ngực:",bg="#79b484").place(x = 180 , y = 130)
lable_cp_loaidaunguc = ['Không có triệu chứng',  'Đau thắt ngực không điển hình', 'Không đau thắt ngực', 'Đau thắt ngực điển hình']  
lable_cp = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_cp_loaidaunguc, state = "readonly")
lable_cp.place(x = 410 , y = 130)
lable_cp.current(0)

lable_trestbps = Label(form,font=("Arial Bold", 10), text = "Huyết áp khi nghỉ ngơi(mm/Hg):",bg="#79b484").place(x = 180 , y = 170)
textbox_trestbps = Entry(form,width = 30,font=("Arial Bold", 10))
textbox_trestbps.place(x = 410 , y = 170)

lable_chol = Label(form,font=("Arial Bold", 10), text = "Cholesterol(mg/dl):",bg="#79b484").place(x = 180 , y = 210)
textbox_chol = Entry(form,width = 30,font=("Arial Bold", 10))
textbox_chol.place(x = 410 , y = 210)

lable_fbs = Label(form,font=("Arial Bold", 10), text = "Lượng đường trong máu: ",bg="#79b484").place(x = 180 , y = 250)
lable_fbs_luongduong = ['<120 mg/dl',  '>120 mg/dl']
lable_fbs = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_fbs_luongduong, state = "readonly")
lable_fbs.place(x = 410 , y = 250)
lable_fbs.current(0)

lable_restecg = Label(form,font=("Arial Bold", 10), text = "Điện tâm đồ khi nghỉ ngơi:",bg="#79b484").place(x = 180 , y = 290)
lable_restecg_dientamdo = ['Bình thường',  'Có sóng ST-T bất thường', 'Phì đại thất trái']
lable_restecg = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_restecg_dientamdo, state = "readonly")
lable_restecg.place(x = 410 , y = 290)
lable_restecg.current(0)

# THÔNG TIN CỘT 2
lable_thalach = Label(form,font=("Arial Bold", 10), text = "Số nhịp đập mỗi phút:",bg="#79b484").place(x = 830 , y = 50)
textbox_thalach = Entry(form,width = 30,font=("Arial Bold", 10))
textbox_thalach.place(x = 1080 , y = 50)

lable_exang = Label(form,font=("Arial Bold", 10), text = "Tập thể dục gây ra đau thắt ngực:",bg="#79b484").place(x = 830 , y = 90)
lable_exang_daunguc = ['Không',  'Có']
lable_exang = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_exang_daunguc, state = "readonly")
lable_exang.place(x = 1080 , y = 90)
lable_exang.current(0)

lable_oldpeak = Label(form,font=("Arial Bold", 10), text = "ST trầm cảm:",bg="#79b484").place(x = 830 , y = 130)
textbox_oldpeak = Entry(form,width = 30,font=("Arial Bold", 10))
textbox_oldpeak.place(x = 1080 , y = 130)

lable_slope = Label(form,font=("Arial Bold", 10), text = "Độ dốc của đoạn ST:",bg="#79b484").place(x = 830 , y = 170)
lable_slope_dodocST = ['Đi xuống',  'Đi lên', 'Cân bằng']
lable_slope = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_slope_dodocST, state = "readonly")
lable_slope.place(x = 1080 , y = 170)
lable_slope.current(0)

lable_ca = Label(form,font=("Arial Bold", 10), text = "Số mạch chính:",bg="#79b484").place(x = 830 , y = 210)
lable_ca_somachchinh = ['0',  '1', '2', '3', '4']
lable_ca = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_ca_somachchinh, state = "readonly")
lable_ca.place(x = 1080 , y = 210)
lable_ca.current(0)

lable_thal = Label(form,font=("Arial Bold", 10), text = "Thalassemia:",bg="#79b484").place(x = 830 , y = 250)
lable_thal_Thalassemia = ['Không',  'Khuyết tật cố định', 'Lưu lượng máu bình thường', 'Khuyết tật có thể đảo ngược']
lable_thal = ttk.Combobox(form,font=("Arial Bold", 10), width = 28, values = lable_thal_Thalassemia, state = "readonly")
lable_thal.place(x = 1080 , y = 250)
lable_thal.current(0)

# KẾT QUẢ DỰ ĐOÁN
lable_people = LabelFrame(form, text = "Kết quả dự đoán", font=("Arial Bold", 13), fg="blue")
lable_people.pack(fill="both", expand="yes")
lable_people.config(bg="#79b484")
# bg="#79b484"


#Khi chỉ sử dụng thuật toán svm
lable_note = Label(form, text = "Khi chưa sử dụng PCA",font=("Arial Bold", 13),fg="blue",bg="#79b484").place(x = 410 , y = 500)


y_svm = svmModel.predict(X_test_main)
lbl3 = Label(form,font=("Arial Bold", 10),bg="#79b484")
lbl3.place(x = 350 , y = 550)
lbl3.configure(text="Tỷ lệ dự đoán đúng của svm: "+str(accuracy_score(y_test_main, y_svm)*100)+"%"+'\n'
                           +"Precision: "+str(precision_score(y_test_main, y_svm)*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test_main, y_svm)*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test_main, y_svm)*100)+"%"+'\n')

#khi dung PCA với thuật toán svm
lable_note = Label(form, text = "Khi sử dụng PCA",font=("Arial Bold", 13),fg="blue",bg="#79b484").place(x = 930 , y = 500)

X_test_PCA_svm = pca_best_svm.transform(X_test_main)
y_svm_PCA = svm_PCA.predict(X_test_PCA_svm)
lbl3 = Label(form,font=("Arial Bold", 10),bg="#79b484")
lbl3.place(x = 850 , y = 550)
lbl3.configure(text="Tỷ lệ dự đoán đúng của svm: "+str(accuracy_score(y_test_main, y_svm_PCA)*100)+"%"+'\n'
                           +"Precision: "+str(precision_score(y_test_main, y_svm_PCA)*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test_main, y_svm_PCA)*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test_main, y_svm_PCA)*100)+"%"+'\n'
                           +"Sử dụng: "+str(num_pca_svm)+"/13 trường dữ liệu")


# Hàm lấy giá từ form
def getValue():
    age = textbox_age.get()
    sex = lable_sex.get()
    if(sex == 'Nam'):
        sex = 1
    else:
        sex = 0
    cp = lable_cp.get()
    if(cp == 'Không có triệu chứng'):
        cp = 0
    elif(cp == 'Đau thắt ngực không điển hình'):
        cp = 1
    elif(cp == 'Không đau thắt ngực'):
        cp = 2
    elif(cp == 'Đau thắt ngực điển hình'):
        cp = 3
    trestbps = textbox_trestbps.get()
    chol = textbox_chol.get()
    fbs = lable_fbs.get()
    if(fbs == '<120 mg/dl'):
        fbs = 0
    elif(fbs == '>120 mg/dl'):
        fbs = 1
    restecg = lable_restecg.get()
    if(restecg == 'Bình thường'):
        restecg = 0
    elif(restecg == 'Có sóng ST-T bất thường'):
        restecg = 1
    elif(restecg == 'Phì đại thất trái'):
        restecg = 2
    thalach = textbox_thalach.get()
    exang = lable_exang.get()
    if(exang == 'Có'):
        exang = 1
    else:
        exang = 0
    oldpeak = textbox_oldpeak.get()
    slope = lable_slope.get()
    if(slope == 'Đi xuống'):
        slope = 0
    elif(slope == 'Đi lên'):
        slope = 1
    elif(slope == 'Cân bằng'):
        slope = 2
    ca = lable_ca.get()

    thal = lable_thal.get()
    if(thal == 'Không'):
        thal = 0
    elif(thal == 'Khuyết tật cố định'):
        thal = 1
    elif(thal == 'Lưu lượng máu bình thường'):
        thal = 2
    elif(thal == 'Khuyết tật có thể đảo ngược'):
        thal = 3

   # gán các giá trị được lấy từ form vào arr 
    X_dudoan = np.array([age, sex, cp, trestbps, chol, fbs,restecg, thalach, exang, oldpeak,slope, ca, thal]).reshape(1, -1)    
    return X_dudoan

# Hàm dự đoán svm
def dudoansvm():
    age = textbox_age.get()
    sex = lable_sex.get()
    cp = lable_cp.get()
    trestbps = textbox_trestbps.get()
    chol = textbox_chol.get()
    fbs = lable_fbs.get()
    restecg = lable_restecg.get()
    thalach = textbox_thalach.get()
    exang = lable_exang.get()
    oldpeak = textbox_oldpeak.get()
    slope = lable_slope.get()
    ca = lable_ca.get()
    thal = lable_thal.get()

    if((age == '') or (sex == '') or (cp == '') or (trestbps == '') or (chol == '') or (fbs == '') or (restecg == '') or (thalach == '') or (exang == '') or (oldpeak == '') or (slope == '') or (ca == '') or (thal == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        if( int(age) < 1 or int(age) > 120):
            messagebox.showerror("Thông báo", "Thông tin tuổi phải từ 0-120")
        elif( int(trestbps) < 0 ) :
            messagebox.showerror("Thông báo", "Thông tin huyết áp khi nghỉ ngơi phải lớn hơn 0")
        elif( int(chol) < 0 ) :
            messagebox.showerror("Thông báo", "Thông tin cholesteron phải lớn hơn 0")
        elif( int(thalach) < 0 ) :
            messagebox.showerror("Thông báo", "Thông tin nhịp tim mỗi phút phải lớn hơn 0")
        elif( float(oldpeak) < 0 ) :
            messagebox.showerror("Thông báo", "Thông tin ST trầm cảm phải lớn hơn 0")
        else:
            X_dudoan = getValue()
            y_kqua = svmModel.predict(X_dudoan)
            if(y_kqua == 1):
                lbl1.configure(text= 'Yes - Bị bệnh')
            else:
                lbl1.configure(text= 'No - Không bị bệnh')

# Button
button_svm = Button(form ,font=("Arial Bold", 10), text = 'Kết quả dự đoán theo svm', command = dudoansvm, background="#04AA6D", foreground="white")
button_svm.place(x = 410 , y = 670)
lbl1 = Label(form, text="",font=("Arial Bold", 10),fg="blue", bg="#79b484")
lbl1.place(x = 410 , y = 700)


# Hàm dự doán PCA + svm
def dudoansvm_PCA():
    age = textbox_age.get()
    sex = lable_sex.get()
    cp = lable_cp.get()
    trestbps = textbox_trestbps.get()
    chol = textbox_chol.get()
    fbs = lable_fbs.get()
    restecg = lable_restecg.get()
    thalach = textbox_thalach.get()
    exang = lable_exang.get()
    oldpeak = textbox_oldpeak.get()
    slope = lable_slope.get()
    ca = lable_ca.get()
    thal = lable_thal.get()

    if((age == '') or (sex == '') or (cp == '') or (trestbps == '') or (chol == '') or (fbs == '') or (restecg == '') or (thalach == '') or (exang == '') or (oldpeak == '') or (slope == '') or (ca == '') or (thal == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = getValue()
        X_new = pca_best_svm.transform(X_dudoan)
        y_kqua = svm_PCA.predict(X_new)
        if(y_kqua == 1):
            lbl1_svmpca.configure(text= 'Yes - Bị bệnh')
        else:
            lbl1_svmpca.configure(text= 'No - Không bị bệnh')

# Button
button_svm_pca = Button(form ,font=("Arial Bold", 10), text = 'Kết quả dự đoán theo svm_PCA', command = dudoansvm_PCA, background="#04AA6D", foreground="white")
button_svm_pca.place(x = 930 , y = 670)
lbl1_svmpca = Label(form, text="",font=("Arial Bold", 10),fg="blue",bg="#79b484")
lbl1_svmpca.place(x = 930 , y = 700)

form.mainloop()
