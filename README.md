### MODEL SELECTION AND EVALUATION FOR CREDIT RISK PREDICTION: A HOME CREDIT CASE STUDY

<img width="1100" height="706" alt="image" src="https://github.com/user-attachments/assets/98b69447-66a6-4195-85d3-bb988b01b82f" />

#### Dataset
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data?source=post_page-----a165466f3c39---------------------------------------)

#### Business Understanding
<p align="justify">
Home Credit merupakan perusahaan keuangan yang menyediakan layanan kredit kepada konsumen yang memiliki keterbatasan akses ke layanan keuangan tradisional. Perusahaannya, menawarkan layanan kredit kepada individu yang tidak memiliki riwayat kredit atau yang berisiko tinggi. Tujuan dari analisis ini yaitu, memahami faktor-faktor yang memengaruhi risiko gagal bayar kredit.

#### Data Understanding
<p align="justify">
Data yang digunakan dalam Final Project ini hanya application_train.
<img width="515" height="413" alt="image" src="https://github.com/user-attachments/assets/bfedfc53-55ff-4d8e-8702-2a662d7660ec" />
<p align="justify">
Berdasarkan Pie Chart, 92% tidak memiliki masalah dalam melunasi pinjaman pada waktu tertentu dan 8% bermasalah, jadi diperlukan analisis lanjutan pada variabel bebas, untuk mengetahui kriteria apa yang menyebabkan nasabah tidak bermasalah ataupun bermasalah dalam membayar kembali pinjamannya.

#### Data Preparation
<img width="488" height="473" alt="image" src="https://github.com/user-attachments/assets/463c815b-50e9-42ad-b6e4-e2459e09ebba" />
<p align="justify">
Data Home Credit awalnya memiliki 307511 rows x 122 columns.

Penanganan Missing Values :
1. Missing values > 40%, dilakukan penghapusan kolom.
2. Missing values yang tersisa diisi dengan median untuk data numerik dan modus untuk data kategorikal.
3. Terdapat data yang memiliki nilai XNA, maka dilakukan handling dengan mengganti data dengan nilai modus.
4. Menghapus kolom FLAG_DOCUMENT karena tidak relevan.
5. Outlier Handling dilakukan menggunakan z-score, data CNT_CHILDREN akan dihapus jika nilai z-score dibawah 3.
6. Terdapat nilai anomali, contohnya terdapat data yang bernilai minus dalam beberapa kolom. Hal ini diperbaiki dengan mengalikan data yang bernilai minus dengan -1.
<p align="justify">
Variabel kategori seringkali tidak dapat diolah secara langsung oleh model atau algoritma machine learning, sehingga dilakukan 2 metode encoding yaitu, One Hot Encoding dilakukan untuk data dengan value unique lebih dari 2 sedangkan sisanya dilakukan label encoder.

### Modelling
Contoh syntax dalam membangun model Logistic Regression:

Sebelum Cross-Validation
```
lr1 = LogisticRegression(max_iter=2000)
lr1.fit(x_over, y_over)
predictions = lr1.predict(x_test)

# print classification report
print(classification_report(y_test, predictions))
confusionmatrix(predictions, y_test)

# Calculate ROC AUC
y_scores = lr1.predict_proba(x_test)[:, 1]  # Menggunakan probabilitas kelas positif
roc_auc_lr1 = roc_auc_score(y_test, y_scores)

# Print ROC AUC
print("ROC AUC:", roc_auc_lr1)
```

Sesudah Cross-Validation
```
# Perform cross-validation and calculate ROC AUC
cv_roc_auc_lr1 = cross_val_score(lr1, x, y, cv=5, scoring='roc_auc')

# Print cross-validated ROC AUC scores
print("Cross-Validated ROC AUC Scores:", cv_roc_auc_lr1)
print("Average Cross-Validated ROC AUC Scores: {:.2f}".format(np.mean(cv_roc_auc_lr1)))
```

#### Evaluation
<img width="647" height="189" alt="image" src="https://github.com/user-attachments/assets/6569fa8e-bc6f-4c68-929d-6dc65cf7a567" />

<p align="justify">
Model klasifikasi terbaik yang didapatkan untuk memprediksi nasabah yang gagal membayar pinjaman pada data Home Credit adalah Model XGBoost.



- Collaboration Project (5 Person)

