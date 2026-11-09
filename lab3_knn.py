
import pandas as pd

columns = [
    "sepal_length", "sepal_width",
    "petal_length", "petal_width", "species"
]

data = pd.read_csv("IrisData_full.csv", header=None, names=columns)

print("Перші 5 записів набору даних Iris:")
print(data.head())

print("\nРозмір датасету:", data.shape)



data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nПерші 5 записів після перемішування:")
print(data.head())




from sklearn.preprocessing import StandardScaler

X = data.drop(columns=["species"])
y = data["species"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\nПерші 5 записів після нормалізації:")
print(X_scaled_df.head())

print("\nСередні значення нормалізованих ознак:")
print(X_scaled_df.mean().round(4))




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nРозділення набору даних:")
print("Кількість навчальних записів:", len(X_train))
print("Кількість тестових записів:", len(X_test))

print("\nРозподіл класів у навчальній вибірці:")
print(y_train.value_counts())

print("\nРозподіл класів у тестовій вибірці:")
print(y_test.value_counts())




from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print("\nТочність класифікації для різних значень K:")

results = []

for k in range(1, 21):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    results.append((k, accuracy))
    print(f"K = {k:2d} | Accuracy = {accuracy:.4f}")




import matplotlib.pyplot as plt

best_k, best_accuracy = max(results, key=lambda item: item[1])

print("\nОптимальне значення K:", best_k)
print(f"Найкраща точність: {best_accuracy * 100:.2f}%")

k_values = [item[0] for item in results]
accuracy_values = [item[1] for item in results]

plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracy_values, marker="o")
plt.axvline(best_k, linestyle="--", label=f"Best K = {best_k}")
plt.xlabel("Кількість сусідів K")
plt.ylabel("Точність класифікації")
plt.title("Залежність точності KNN від значення K")
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("lab3_accuracy.png", dpi=200)
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train_scaled, y_train)

y_pred_best = best_model.predict(X_test_scaled)

print("\nЗвіт про якість класифікації:")
print(classification_report(y_test, y_pred_best))

print("\nМатриця помилок:")
print(confusion_matrix(y_test, y_pred_best))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_best,
    cmap="Blues"
)

plt.title("Матриця помилок KNN")
plt.tight_layout()
plt.savefig("lab3_confusion.png", dpi=200)
plt.show()
