
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

BASE_DIR = os.path.dirname(__file__)
# -------------------------------------------------
# Load data dan model (bisa modifikasi sesuai kebutuhan)
@st.cache_data
def load_data():
    df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')
    return df

kmeans_model = joblib.load(os.path.join(BASE_DIR, 'kmeans_model.pkl'))
logreg_model = joblib.load(os.path.join(BASE_DIR, 'logreg_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler_logreg.pkl'))

# -------------------------------------------------
# Fungsi preprocessing & clustering
@st.cache_data
def preprocess_and_cluster(df):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.cluster import KMeans

    # Pilih fitur untuk clustering
    unsupervised_df = df[['Total Purchase Amount', 'Product Price', 'Customer Age', 'Quantity']].copy()

    # Remove outliers IQR
    def remove_outliers_iqr(dataframe, columns):
        for col in columns:
            Q1 = dataframe[col].quantile(0.25)
            Q3 = dataframe[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            dataframe = dataframe[(dataframe[col] >= lower_bound) & (dataframe[col] <= upper_bound)]
        return dataframe

    unsupervised_df = remove_outliers_iqr(unsupervised_df, unsupervised_df.columns)

    # Scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(unsupervised_df)

    # KMeans clustering
    kmeans = joblib.load('kmeans_model.pkl')
    clusters = kmeans.predict(scaled_features)
    unsupervised_df['Spending Cluster'] = clusters

    # Gabungkan kembali ke df asli sesuai index
    df = df.loc[unsupervised_df.index]
    df['Spending Cluster'] = clusters

    return df, kmeans

# -------------------------------------------------
# Fungsi untuk model prediksi churn (gunakan hasil best_model)
@st.cache_data
def load_model_and_predict(df):
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score

    supervised_df = df[['Total Purchase Amount', 'Product Price', 'Customer Age', 'Gender','Returns', 'Churn']].dropna()
    le = LabelEncoder()
    supervised_df['Gender'] = le.fit_transform(supervised_df['Gender'])
    X = supervised_df[['Total Purchase Amount', 'Product Price', 'Customer Age', 'Gender','Returns' ]]
    y = supervised_df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = joblib.load('scaler_logreg.pkl')
    logreg = joblib.load('logreg_model.pkl')

    X_test_scaled = scaler.transform(X_test)

    y_pred = logreg.predict(X_test_scaled)
    y_proba = logreg.predict_proba(X_test_scaled)[:,1]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': logreg.score(X_test_scaled, y_test),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'X_test': X_test
    }

    return metrics

# -------------------------------------------------
# Streamlit Dashboard UI

st.title("Dashboard Analisis dan Prediksi Pelanggan E-Commerce")

# Load data
df = load_data()

# Sidebar filter
st.sidebar.header("Filter Data")
gender_filter = st.sidebar.multiselect("Gender", options=df['Gender'].unique(), default=df['Gender'].unique())
payment_filter = st.sidebar.multiselect("Metode Pembayaran", options=df['Payment Method'].unique() if 'Payment Method' in df.columns else [], default=df['Payment Method'].unique() if 'Payment Method' in df.columns else [])
category_filter = st.sidebar.multiselect("Kategori Produk", options=df['Product Category'].unique() if 'Product Category' in df.columns else [], default=df['Product Category'].unique() if 'Product Category' in df.columns else [])

filtered_df = df[
    (df['Gender'].isin(gender_filter)) & 
    ((df['Payment Method'].isin(payment_filter)) if 'Payment Method' in df.columns else True) &
    ((df['Product Category'].isin(category_filter)) if 'Product Category' in df.columns else True)
]

st.header("Ringkasan Data")
total_customers = filtered_df.shape[0]
avg_purchase = filtered_df['Total Purchase Amount'].mean()
churn_counts = filtered_df['Churn'].value_counts(normalize=True) if 'Churn' in filtered_df.columns else None

st.markdown(f"- **Jumlah Total Pelanggan:** {total_customers}")
st.markdown(f"- **Rata-rata Total Pembelian:** Rp {avg_purchase:,.2f}")
if churn_counts is not None:
    st.markdown(f"- **Rasio Pelanggan Churn:** {churn_counts.get(1,0)*100:.2f}%")
    st.markdown(f"- **Rasio Pelanggan Tidak Churn:** {churn_counts.get(0,0)*100:.2f}%")

# Segmentasi Pelanggan
st.header("Segmentasi Pelanggan (Clustering)")

df_clustered, kmeans = preprocess_and_cluster(filtered_df)

# Scatter plot cluster
fig, ax = plt.subplots(figsize=(10,6))
sns.scatterplot(data=df_clustered, x='Customer Age', y='Total Purchase Amount', hue='Spending Cluster', palette='tab10', ax=ax)
ax.set_title("Segmentasi Pelanggan Berdasarkan Usia dan Total Pembelian")
st.pyplot(fig)

# Deskripsi segmen
segmen_desc = df_clustered.groupby('Spending Cluster').agg({
    'Total Purchase Amount': 'mean',
    'Customer Age': 'mean',
    'Quantity': 'mean'
}).rename(columns={
    'Total Purchase Amount': 'Rata-rata Total Pembelian',
    'Customer Age': 'Rata-rata Usia',
    'Quantity': 'Rata-rata Kuantitas'
})
st.dataframe(segmen_desc)

# Distribusi usia dan pengeluaran per segmen
fig2, axes = plt.subplots(1, 2, figsize=(15,5))
sns.boxplot(x='Spending Cluster', y='Customer Age', data=df_clustered, ax=axes[0])
axes[0].set_title('Distribusi Usia per Segmen')
sns.boxplot(x='Spending Cluster', y='Total Purchase Amount', data=df_clustered, ax=axes[1])
axes[1].set_title('Distribusi Total Pembelian per Segmen')
st.pyplot(fig2)

# Analisis Loyalitas Pelanggan
st.header("Analisis Loyalitas Pelanggan (Prediksi Churn)")

metrics = load_model_and_predict(filtered_df)

# Tabel prediksi churn dengan probabilitas (ambil sampel)
prediksi_df = pd.DataFrame({
    'Actual': metrics['y_test'],
    'Predicted': metrics['y_pred'],
    'Probabilitas Churn': metrics['y_proba']
}).reset_index(drop=True)
st.dataframe(prediksi_df.head(20))

# Visualisasi Confusion Matrix
fig3, ax3 = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=metrics['confusion_matrix'], display_labels=['Tidak Churn', 'Churn'])
disp.plot(ax=ax3)
st.pyplot(fig3)

# Grafik metrik evaluasi
st.subheader("Metrik Evaluasi Model")
metrics_df = pd.DataFrame({
    'Metrik': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Nilai': [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
})

fig4, ax4 = plt.subplots()
sns.barplot(x='Metrik', y='Nilai', data=metrics_df, ax=ax4)
ax4.set_ylim(0,1)
ax4.set_title("Perbandingan Metrik Evaluasi Model")
st.pyplot(fig4)

# Insight Interaktif: Timeline pembelian
if 'Order Date' in filtered_df.columns:
    st.header("Tren Pembelian Berdasarkan Waktu")
    filtered_df['Order Date'] = pd.to_datetime(filtered_df['Order Date'])
    timeline = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Total Purchase Amount'].sum()
    timeline.index = timeline.index.to_timestamp()
    st.line_chart(timeline)

# Return rate berdasarkan segmen dan gender (jika tersedia)
if 'Returns' in filtered_df.columns and 'Spending Cluster' in filtered_df.columns and 'Gender' in filtered_df.columns:
    st.header("Return Rate Berdasarkan Segmen dan Gender")
    return_rate = filtered_df.groupby(['Spending Cluster', 'Gender'])['Returns'].mean().reset_index()
    fig5, ax5 = plt.subplots()
    sns.barplot(x='Spending Cluster', y='Returns', hue='Gender', data=return_rate, ax=ax5)
    ax5.set_title("Rata-rata Return per Segmen dan Gender")
    st.pyplot(fig5)
