"""
Корреляционный анализ и кластеризация данных опроса студентов.
Переписано с R на Python.
Автор: Пётр
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2_contingency

# Отключаем предупреждения для чистоты вывода
warnings.filterwarnings("ignore")


# ============================================================
# 1. Загрузка данных из Excel-файла (два листа)
# ============================================================

FILE_PATH = "Data_040426_v2.xlsx"

# Лист "Data" — основной датасет, пропускаем первую строку заголовков
data = pd.read_excel(FILE_PATH, sheet_name="Data", skiprows=1)

# Лист "Data_2" — дополнительный (расширенный) датасет
data_am = pd.read_excel(FILE_PATH, sheet_name="Data_2", skiprows=1)


# ============================================================
# 2. Преобразование строковых столбцов в категориальные коды
# ============================================================

def strings_to_factors(df):
    """Конвертирует все строковые столбцы в категориальный тип (аналог factor в R)."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    return df

data = strings_to_factors(data)
data_am = strings_to_factors(data_am)


# ============================================================
# 3. Функция для подсчёта пропусков (NA) по столбцам
# ============================================================

def na_report(df):
    """
    Возвращает DataFrame с информацией о пропусках:
    — название переменной
    — абсолютная частота пропусков
    — относительная частота пропусков
    """
    na_counts = df.isnull().sum()
    na_counts = na_counts[na_counts > 0].sort_values()
    report = pd.DataFrame({
        "variable": na_counts.index,
        "freq_abs": na_counts.values,
        "freq_rel": (na_counts.values / len(df)).round(4)
    })
    return report

print("=== Пропуски до очистки (data) ===")
print(na_report(data), "\n")


# ============================================================
# 4. Удаление столбцов и строк с пропусками
# ============================================================

# Столбцы, которые полностью (или почти) пустые — удаляем
DROP_COLS = [
    "Вопрос3", "Вопрос4", "Вопрос6",
    "Вопрос15", "Вопрос16", "Вопрос32",
    "Вопрос36", "Вопрос46", "Вопрос47"
]

data = data.drop(columns=DROP_COLS, errors="ignore")
data_am = data_am.drop(columns=DROP_COLS, errors="ignore")

print("=== Пропуски после удаления столбцов (data) ===")
print(na_report(data), "\n")

# Удаляем оставшиеся строки с пропусками
data = data.dropna()
data_am = data_am.dropna()

print("=== Пропуски после удаления строк (data) ===")
print(na_report(data), "\n")


# ============================================================
# 5. Описательные статистики данных
# ============================================================

print("=== Описательные статистики ===")
print(data.describe(include="all"), "\n")


# ============================================================
# 6. Подготовка числовых данных для корреляционной матрицы
# ============================================================

# Берём столбцы с 9-го по 47-й (нумерация с нуля — индексы 8:47)
data_num = data.iloc[:, 8:47].copy()
data_num2 = data_am.iloc[:, 8:47].copy()

# Конвертируем категориальные переменные в их числовые коды
for col in data_num.select_dtypes(include="category").columns:
    data_num[col] = data_num[col].cat.codes + 1  # +1, чтобы совпадало с R (factor → numeric)

for col in data_num2.select_dtypes(include="category").columns:
    data_num2[col] = data_num2[col].cat.codes + 1


# ============================================================
# 7. Корреляционная матрица и тепловые карты
# ============================================================

# Рассчитываем корреляцию Пирсона
correl = data_num.corr()
correl2 = data_num2.corr()

def plot_heatmap(corr_matrix, title):
    """Строит тепловую карту корреляционной матрицы."""
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr_matrix,
        vmin=-1, vmax=1,
        cmap=sns.diverging_palette(240, 10, as_cmap=True),  # blue-white-red
        square=True,
        linewidths=0.3,
        ax=ax
    )
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(title.replace(" ", "_") + ".png", dpi=150)
    plt.show()

plot_heatmap(correl, "Корреляционная матрица")
plot_heatmap(correl2, "Корреляционная матрица 2")


# ============================================================
# 8. Хи-квадрат тесты (связь «Оценки» с другими вопросами)
# ============================================================

def chi_square_test(df, col1, col2):
    """
    Строит таблицу сопряжённости и проводит тест хи-квадрат.
    Выводит таблицу и результат теста.
    """
    ct = pd.crosstab(df[col1], df[col2])
    print(f"\n--- Таблица сопряжённости: {col1} × {col2} ---")
    print(ct)

    chi2, p_value, dof, expected = chi2_contingency(ct)
    print(f"Хи-квадрат = {chi2:.4f},  p-value = {p_value:.4f},  df = {dof}")
    return chi2, p_value, dof

# Список вопросов для проверки связи с «Оценки» на основном датасете
questions_main = [
    "Вопрос2", "Вопрос10", "Вопрос11", "Вопрос19",
    "Вопрос20", "Вопрос22", "Вопрос31", "Вопрос33", "Вопрос43"
]

print("\n========== Хи-квадрат тесты (data_num) ==========")
for q in questions_main:
    if q in data_num.columns:
        chi_square_test(data_num, "Оценки", q)


# ============================================================
# 9. Столбчатые диаграммы (stacked bar charts) — доли по категориям
# ============================================================

# Цвета для уровней оценок
GRADE_COLORS = {
    "Преимущественно удовлетворительно (средний балл до 3,49)": "#d73027",
    "Преимущественно хорошо (средний балл от 3,5 до 4,49)": "#fc8d59",
    "Преимущественно отлично (средний балл от 4,5 до 4,74)": "#91cf60",
    "Круглый отличник (средний балл от 4,75)": "#1a9850"
}

def stacked_bar_chart(df, x_col, fill_col, xlabel, title="Доли внутри категорий"):
    """
    Строит нормированную столбчатую диаграмму с горизонтальными полосами.
    Аналог geom_bar(position='fill') + coord_flip() в ggplot2.
    """
    ct = pd.crosstab(df[x_col], df[fill_col], normalize="index")

    # Порядок столбцов по ключам словаря (от низшей к высшей оценке)
    ordered = [c for c in GRADE_COLORS if c in ct.columns]
    ct = ct.reindex(columns=ordered, fill_value=0)

    colors = [GRADE_COLORS[c] for c in ct.columns]

    fig, ax = plt.subplots(figsize=(10, max(4, len(ct) * 0.5)))
    ct.plot.barh(stacked=True, color=colors, ax=ax, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Доля")
    ax.set_ylabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)

    # Формат оси X в процентах
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(title="Оценки", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(f"bar_{x_col}.png", dpi=150)
    plt.show()

# --- Графики для расширенного датасета (data_am) ---
bar_configs_am = [
    ("Вопрос2",  "Готовлюсь с однокурсниками?"),
    ("Вопрос33", "Знакомства"),
    ("Вопрос45", "Трачу много времени на учебу"),
]

for col, label in bar_configs_am:
    if col in data_am.columns and "Оценки" in data_am.columns:
        stacked_bar_chart(data_am, col, "Оценки", label)

# Хи-квадрат тесты для расширенного датасета
print("\n========== Хи-квадрат тесты (data_num2) ==========")
for col, _ in bar_configs_am:
    if col in data_num2.columns:
        chi_square_test(data_num2, "Оценки", col)

# --- Графики для основного датасета (data) ---
bar_configs_main = [
    ("Вопрос19", "Развитие проф навыков"),
    ("Вопрос10", "Объединение студентов в группы"),
    ("Вопрос11", "Помощь между одногруппниками"),
    ("Вопрос22", "Связь между учебой и профессией"),
    ("Вопрос31", "Информированность о мероприятиях"),
    ("Вопрос33", "Связь между мероприятиями и знакомствами"),
    ("Вопрос43", "Оценки круга общения"),
    ("Вопрос28", "Оценки круга общения"),
]

for col, label in bar_configs_main:
    if col in data.columns and "Оценки" in data.columns:
        stacked_bar_chart(data, col, "Оценки", label)

# Дополнительный хи-квадрат тест для Вопрос28
if "Вопрос28" in data_num.columns:
    chi_square_test(data_num, "Оценки", "Вопрос28")


# ============================================================
# 10. Кластерный анализ — первый набор (все переменные)
# ============================================================

# Столбцы с 10-го по 47-й (индексы 9:47)
data_clust = data.iloc[:, 9:47].copy()

# Конвертируем категориальные столбцы в числа
for col in data_clust.select_dtypes(include="category").columns:
    data_clust[col] = data_clust[col].cat.codes + 1

# Нормируем данные делением на максимум (min-max с нулевым минимумом)
data_clust_norm = data_clust / data_clust.max()


# ============================================================
# 11. Иерархическая кластеризация и дендрограммы (набор 1)
# ============================================================

# Вычисляем матрицу расстояний (Манхэттен)
dist_manhattan = pdist(data_clust_norm, metric="cityblock")

# Дендрограмма 1 — метод полной связи (complete linkage)
linkage_complete = linkage(dist_manhattan, method="complete")
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(linkage_complete, no_labels=True, ax=ax)
ax.set_title("Дендрограмма 1, метод complete")
plt.tight_layout()
plt.savefig("dendro_complete_1.png", dpi=150)
plt.show()

# Дендрограмма 2 — метод Уорда (ward)
linkage_ward = linkage(dist_manhattan, method="ward")
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(linkage_ward, no_labels=True, ax=ax)
ax.axhline(y=22, color="red", linestyle="--", label="h=22")
ax.axhline(y=17, color="violet", linestyle="--", label="h=17")
ax.set_title("Дендрограмма 2, метод ward.D")
ax.legend()
plt.tight_layout()
plt.savefig("dendro_ward_1.png", dpi=150)
plt.show()

# Дендрограмма 3 — метод центроидов (centroid)
linkage_centroid = linkage(dist_manhattan, method="centroid")
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(linkage_centroid, no_labels=True, ax=ax)
ax.set_title("Дендрограмма 3, метод centroid")
plt.tight_layout()
plt.savefig("dendro_centroid_1.png", dpi=150)
plt.show()


# ============================================================
# 12. Профили кластеров (средние значения по кластерам)
# ============================================================

def print_cluster_profiles(linkage_matrix, cluster_data, n_clusters, id_series):
    """
    Разбивает наблюдения на кластеры и выводит:
    — номера наблюдений в каждом кластере
    — средние значения переменных по кластерам (аналог barchart в R)
    """
    labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")

    print(f"\n=== Профили кластеров (k={n_clusters}) ===")
    for i in range(1, n_clusters + 1):
        mask = labels == i
        members = id_series[mask].tolist()
        print(f"Cluster {i}: {members}")

    # Средние значения переменных по кластерам
    cluster_data_copy = cluster_data.copy()
    cluster_data_copy["cluster"] = labels
    profiles = cluster_data_copy.groupby("cluster").mean()
    print("\nСредние значения по кластерам:")
    print(profiles.round(2))
    return labels

# Кластеризация методом Уорда — k=4 и k=7
print_cluster_profiles(linkage_ward, data_clust, 4, data["No"])
print_cluster_profiles(linkage_ward, data_clust, 7, data["No"])


# ============================================================
# 13. Кластерный анализ — второй набор (выбранные переменные)
# ============================================================

# Выбираем конкретные столбцы по индексам (аналог R: data_am[,c(10,11,23,32,33,36,47)])
selected_cols_idx = [9, 10, 22, 31, 32, 35, 46]
data_clust2 = data_am.iloc[:, selected_cols_idx].copy()

# Конвертируем категории в числа
for col in data_clust2.select_dtypes(include="category").columns:
    data_clust2[col] = data_clust2[col].cat.codes + 1

# Нормируем данные
data_clust_norm2 = data_clust2 / data_clust2.max()


# ============================================================
# 14. Иерархическая кластеризация и дендрограммы (набор 2)
# ============================================================

dist_manhattan_2 = pdist(data_clust_norm2, metric="cityblock")

# Дендрограмма — метод полной связи
linkage_complete_2 = linkage(dist_manhattan_2, method="complete")
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(linkage_complete_2, no_labels=True, ax=ax)
ax.set_title("Дендрограмма 1 (набор 2), метод complete")
plt.tight_layout()
plt.savefig("dendro_complete_2.png", dpi=150)
plt.show()

# Дендрограмма — метод Уорда
linkage_ward_2 = linkage(dist_manhattan_2, method="ward")
fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(linkage_ward_2, no_labels=True, ax=ax)
ax.set_title("Дендрограмма 2 (набор 2), метод ward.D")
plt.tight_layout()
plt.savefig("dendro_ward_2.png", dpi=150)
plt.show()

# Кластеризация — k=3
print_cluster_profiles(linkage_ward_2, data_clust2, 3, data_am["No"])


# ============================================================
# 15. Завершение
# ============================================================
print("\n=== Анализ завершён. Графики сохранены в текущей директории. ===")
