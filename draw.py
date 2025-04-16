import matplotlib.pyplot as plt

# Danh sách các thuật toán
algorithms = ['K-Medoids', 'CURE', 'DBSCAN']

# Hàm để nhập và xác thực dữ liệu cho một chỉ số
def get_metric_scores(metric_name):
    while True:
        try:
            scores_input = input(f"Nhập {metric_name} cho k-medoids, CURE, DBSCAN (cách nhau bằng khoảng trắng): ")
            scores = [float(x) for x in scores_input.split()]
            if len(scores) != 3:
                raise ValueError(f"{metric_name}: Vui lòng nhập đúng 3 giá trị số.")
            return scores
        except ValueError as e:
            print(e)

# Nhập dữ liệu cho từng chỉ số
print("Nhập dữ liệu để vẽ biểu đồ:")
silhouette_scores = get_metric_scores("Silhouette Scores")
db_scores = get_metric_scores("Davies-Bouldin Indices")
ch_scores = get_metric_scores("Calinski-Harabasz Indices")

# Tạo figure và các bảng con
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Vẽ biểu đồ cho Silhouette Score
ax1.bar(algorithms, silhouette_scores, color='blue')
ax1.set_title('Silhouette Score')
ax1.grid(True, axis='y')

# Vẽ biểu đồ cho Davies-Bouldin Index
ax2.bar(algorithms, db_scores, color='orange')
ax2.set_title('Davies-Bouldin Index')
ax2.grid(True, axis='y')

# Vẽ biểu đồ cho Calinski-Harabasz Index
ax3.bar(algorithms, ch_scores, color='green')
ax3.set_title('Calinski-Harabasz Index')
ax3.grid(True, axis='y')

# Điều chỉnh bố cục và lưu biểu đồ
plt.tight_layout()
plt.savefig('clustering_comparison.png')
plt.show()