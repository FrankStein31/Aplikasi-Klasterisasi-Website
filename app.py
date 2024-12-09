from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import os
import time
import matplotlib
matplotlib.use('Agg')  # Menggunakan backend non-GUI untuk mencegah error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import chardet
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'virnandaheidylalala'  # Set secret key untuk flash messages
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['IMAGE_FOLDER'] = 'static/images'

# Pastikan folder untuk uploads dan images ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
if not os.path.exists(app.config['IMAGE_FOLDER']):
    os.makedirs(app.config['IMAGE_FOLDER'])

# Variable global untuk memeriksa status upload
current_file = None  # Variabel untuk menyimpan file yang diupload

@app.route('/')
def index():
    return redirect(url_for('upload'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global current_file
    if request.method == 'POST':
        file = request.files['dataset']
        if file.filename == '':
            flash("Tidak ada file yang dipilih. Silakan pilih file untuk diunggah.", 'error')
            return redirect(url_for('upload'))
        if not file.filename.endswith('.csv'):
            flash("Hanya file CSV yang diperbolehkan.", 'error')
            return redirect(url_for('upload'))
        

        # Simpan file yang diupload
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Deteksi encoding
        try:
            with open(filepath, 'rb') as f:
                result = chardet.detect(f.read(10000))
                detected_encoding = result['encoding']

            # Baca file dengan encoding terdeteksi
            data = pd.read_csv(filepath, encoding=detected_encoding)
        except UnicodeDecodeError:
            flash("Encoding tidak didukung. Coba unggah file dengan encoding UTF-8 atau Latin1.", 'error')
            return redirect(url_for('upload'))
        except Exception as e:
            flash(f"Terjadi kesalahan saat membaca file: {str(e)}", 'error')
            return redirect(url_for('upload'))

        # Simpan nama file
        current_file = file.filename

        # Cek missing values
        if data.isnull().values.any():
            missing_info = data.isnull().sum().to_dict()
            return render_template(
                'handle_missing.html',
                filename=file.filename,
                data=data.to_html(classes='table table-striped', na_rep="MISSING"),  # Highlight missing values
                missing_info=missing_info
            )
        
        return render_template(
            'data_overview.html',
            filename=file.filename,
            data=data.to_html(classes='table table-striped')
        )

    return render_template('upload.html')

@app.route('/handle_missing/<filename>', methods=['GET', 'POST'])
def handle_missing(filename):
    global current_file
    if current_file is None:
        flash("Silakan unggah dataset terlebih dahulu.", 'error')
        return redirect(url_for('upload'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        flash(f"Terjadi kesalahan saat membaca file: {str(e)}", 'error')
        return redirect(url_for('upload'))

    # Menambahkan indikator missing value
    highlighted_data = data.copy()
    for col in highlighted_data.columns:
        highlighted_data[col] = highlighted_data[col].apply(
            lambda x: f"<span style='color: red;'>MISSING</span>" if pd.isnull(x) else x
        )

    if request.method == 'POST':
        # Pilihan metode dari form
        method = request.form.get('method', None)

        if not method:
            flash("Silakan pilih metode untuk menangani missing values.", 'error')
            return redirect(url_for('handle_missing', filename=filename))

        try:
            if method == 'drop_rows':
                data = data.dropna()  # Hapus baris dengan missing values
            elif method == 'drop_columns':
                data = data.dropna(axis=1)  # Hapus kolom dengan missing values
            elif method == 'fill_mean':
                data = data.fillna(data.mean(numeric_only=True))  # Isi dengan mean
            elif method == 'fill_median':
                data = data.fillna(data.median(numeric_only=True))  # Isi dengan median
            elif method == 'fill_mode':
                data = data.fillna(data.mode().iloc[0])  # Isi dengan mode
            else:
                flash("Metode tidak valid.", 'error')
                return redirect(url_for('handle_missing', filename=filename))

            # Simpan dataset yang telah diperbaiki
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
            data.to_csv(processed_filepath, index=False)

            return render_template(
                'data_overview.html',
                filename=filename,
                data=data.to_html(classes='table table-striped'),
                processed=True  # Menunjukkan bahwa data telah diproses
            )

        except Exception as e:
            flash(f"Terjadi kesalahan saat menangani missing values: {str(e)}", 'error')
            return redirect(url_for('handle_missing', filename=filename))

    # Jika GET, tampilkan informasi missing values dan data asli dengan highlight
    missing_info = data.isnull().sum()
    return render_template(
        'handle_missing.html',
        filename=filename,
        missing_info=missing_info.to_dict(),
        data_before=highlighted_data.to_html(classes='table table-striped', escape=False),
        data_after=None,
        processed=False
    )

@app.route('/data_overview')
def data_overview():
    if current_file is None:
        flash("Silakan unggah dataset terlebih dahulu.", 'error')
        return redirect(url_for('upload'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], current_file)
    
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        flash(f"Terjadi kesalahan saat membaca file: {str(e)}", 'error')
        return redirect(url_for('upload'))
    
    # Proses data sesuai metode yang dipilih di form
    method = request.args.get('method', 'drop_rows')  # Default method 'drop_rows'

    # Proses data dengan menggunakan handle_missing (perbaiki agar dapat mengembalikan data yang diproses)
    data = handle_missing(data, method)

    if data is None:
        # Jika terjadi kesalahan saat pemrosesan data
        return redirect(url_for('handle_missing', filename=filename))
    
    return render_template('data_overview.html', data=data.to_html(classes='table table-striped'), filename=current_file)

@app.route('/elbow/<filename>', methods=['GET', 'POST'])
def elbow(filename=None):
    global current_file
    filename = filename or current_file
    if filename is None:
        flash("Silakan unggah dataset terlebih dahulu.", 'error')
        return redirect(url_for('upload'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        flash(f"Terjadi kesalahan saat membaca file: {str(e)}", 'error')
        return redirect(url_for('upload'))

    method = request.args.get('method', 'drop_rows')
    data = handle_missing(data, method)

    # Ambil hanya kolom numerik
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    columns = numeric_data.columns.tolist()

    # Ambil kolom yang dipilih (jika ada)
    selected_columns = request.form.getlist('selected_columns') or columns[:2]  # Default: 2 kolom pertama
    if len(selected_columns) < 2:
        flash("Silakan pilih setidaknya dua kolom untuk analisis elbow.", 'error')
        return redirect(url_for('upload'))

    # Proses dataset dengan kolom terpilih
    numeric_data = numeric_data[selected_columns]

    # Standardisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_data)

    # Hitung inertia untuk berbagai jumlah klaster
    inertias = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # Temukan elbow menggunakan KneeLocator
    knee_locator = KneeLocator(K, inertias, curve="convex", direction="decreasing")
    optimal_k = knee_locator.knee

    # Buat grafik elbow
    elbow_path = os.path.join(app.config['IMAGE_FOLDER'], 'elbow_plot.png')
    plt.figure(figsize=(8, 6))
    plt.plot(K, inertias, marker='o', linestyle='--')
    plt.title('Metode Elbow untuk Menentukan Jumlah Klaster Optimal')
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('Inertia')
    plt.savefig(elbow_path)
    plt.close()

    return render_template(
        'elbow.html',
        filename=filename,
        elbow_path=url_for('static', filename='images/elbow_plot.png'),
        optimal_k=optimal_k,
        columns=columns,
        selected_columns=selected_columns
    )

@app.route('/clustering/<filename>', methods=['POST'])
def clustering(filename):
    if current_file is None:
        flash("Silakan unggah dataset terlebih dahulu.", 'error')
        return redirect(url_for('upload'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        flash(f"Terjadi kesalahan saat membaca file: {str(e)}", 'error')
        return redirect(url_for('upload'))

    method = request.args.get('method', 'drop_rows')
    data = handle_missing(data, method)

    # Ambil kolom yang dipilih
    selected_columns = request.form.getlist('selected_columns')
    if not selected_columns or len(selected_columns) < 2:
        flash("Silakan pilih setidaknya dua kolom untuk proses klasterisasi.", 'error')
        return redirect(url_for('upload'))

    # Filter dataset dengan kolom terpilih
    data = data[selected_columns]

    # Standardisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # Ambil nilai k dari form
    try:
        k = int(request.form['k'])
    except ValueError:
        flash("Masukkan jumlah klaster yang valid.", 'error')
        return redirect(url_for('upload'))

    # Klasterisasi menggunakan KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    data['Cluster'] = clusters

    # Simpan data per klaster
    cluster_data = defaultdict(list)
    for cluster in range(k):
        cluster_data[cluster] = data[data['Cluster'] == cluster]

    # Buat grafik hasil clustering
    cluster_path = os.path.join(app.config['IMAGE_FOLDER'], 'cluster_plot.png')
    plt.figure(figsize=(8, 6))
    plt.scatter(
        data.iloc[:, 0],
        data.iloc[:, 1],
        c=clusters,
        cmap='viridis',
        alpha=0.6
    )
    plt.title(f'Klasterisasi dengan {k} Klaster')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.savefig(cluster_path)
    plt.close()

    return render_template(
        'result.html',
        cluster_path=url_for('static', filename='images/cluster_plot.png'),
        cluster_data=cluster_data,
        filename=filename
    )

@app.route('/visualize/<filename>')
def visualize(filename):
    if current_file is None:
        flash("Silakan unggah dataset terlebih dahulu.", 'error')
        return redirect(url_for('upload'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)

    # Select numeric columns and drop the 'Year' column if it exists
    if 'Year' in data.columns:
        numeric_data = data.select_dtypes(include=['float64', 'int64']).drop(columns=['Year'])
    else:
        numeric_data = data.select_dtypes(include=['float64', 'int64'])

    if numeric_data.empty:
        flash("Dataset tidak memiliki kolom numerik untuk visualisasi.", 'error')
        return redirect(url_for('upload'))

    # Calculate descriptive statistics (mean, median, std, min, max)
    stats = numeric_data.describe().T
    stats['median'] = numeric_data.median()

    # Convert stats to a dictionary for easier access in template
    stats_dict = stats.to_dict(orient='index')

    # Prepare for visualization
    visualize_folder = os.path.join(app.config['IMAGE_FOLDER'], 'visualize')
    if not os.path.exists(visualize_folder):
        os.makedirs(visualize_folder)

    # Hapus file sebelumnya di folder visualize
    for file in os.listdir(visualize_folder):
        os.remove(os.path.join(visualize_folder, file))

    plots = [] 

    # Generate Histogram plots for each column
    for column in numeric_data.columns:
        sanitized_column = column.replace("(", "_").replace(")", "_").replace("\\", "_").replace(" ", "_").replace("/", "_")

        plt.figure(figsize=(6, 4))
        sns.histplot(data[column], kde=True, color=sns.color_palette("Set2")[0], bins=20)  # Change color
        plt.title(f'Histogram: {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plot_path = os.path.join(visualize_folder, f'{sanitized_column}_hist.png')
        plt.savefig(plot_path)
        plt.close()
        plots.append(url_for('static', filename=f'images/visualize/{sanitized_column}_hist.png'))

    # Pairwise Scatter Plot (if more than one numeric column)
    if len(numeric_data.columns) >= 2:
        plt.figure(figsize=(8, 6))
        sns.pairplot(numeric_data, diag_kind='kde', plot_kws={'alpha': 0.6})
        scatter_path = os.path.join(visualize_folder, 'scatter_plot.png')
        plt.savefig(scatter_path)
        plt.close()
        plots.append(url_for('static', filename='images/visualize/scatter_plot.png'))

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = numeric_data.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    heatmap_path = os.path.join(visualize_folder, 'heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    plots.append(url_for('static', filename='images/visualize/heatmap.png'))

    # Generate Pie Chart for categorical columns (if exists)
    categorical_data = data.select_dtypes(include=['object', 'category'])
    for column in categorical_data.columns:
        sanitized_column = column.replace("(", "_").replace(")", "_").replace("\\", "_").replace(" ", "_").replace("/", "_")
        
        # Calculate value counts
        value_counts = data[column].value_counts().head(5)

        explode = [0.1 if i == value_counts.idxmax() else 0 for i in range(len(value_counts))]
        
        # Plot pie chart
        plt.figure(figsize=(6, 6))
        value_counts.plot.pie(autopct='%1.1f%%', colors=sns.color_palette("Set3", len(value_counts)), startangle=90, explode=explode)
        plt.title(f'Pie Chart: {column}')
        pie_path = os.path.join(visualize_folder, f'{sanitized_column}_pie.png')
        plt.savefig(pie_path)
        plt.close()
        plots.append(url_for('static', filename=f'images/visualize/{sanitized_column}_pie.png'))

    return render_template('visualize.html', plots=plots, columns=numeric_data.columns.tolist(), stats=stats_dict)

def handle_missing(data, method):
    if method == 'drop_rows':
        return data.dropna()
    elif method == 'drop_columns':
        return data.dropna(axis=1)
    elif method == 'fill_mean':
        return data.fillna(data.mean(numeric_only=True))
    elif method == 'fill_median':
        return data.fillna(data.median(numeric_only=True))
    elif method == 'fill_mode':
        return data.fillna(data.mode().iloc[0])
    else:
        # Jika metode tidak valid
        return None

if __name__ == '__main__':
    app.run(debug=True) 
