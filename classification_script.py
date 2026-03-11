import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib


def run_classification(file_path):

    print(f"Wczytywanie danych z {file_path}...")
    df = pd.read_csv(file_path)

    # Preprocessing
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['growth direction'])
    class_names = le.classes_

    X = df.drop(['growth direction', 'target'], axis=1)
    y = df['target']
    feature_names = X.columns.tolist()

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Trening modelu
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Zapisywanie modelu i pomocniczych plików
    joblib.dump(clf, 'model_orto.joblib')
    joblib.dump(le, 'encoder_klas.joblib')
    joblib.dump(feature_names, 'lista_cech.joblib')
    print("Zapisano model do pliku: model_orto.joblib")

    # Predykcja i Ewaluacja
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)

    print(f"\nDokładność (Accuracy): {accuracy:.4f}")
    print("\nRaport klasyfikacji:")
    print(report)

    # Wizualizacja
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Przewidziane')
    plt.ylabel('Rzeczywiste')
    plt.title('Macierz pomyłek')
    plt.savefig('confusion_matrix.png')

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Istotność cech w modelu")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')



if __name__ == "__main__":
    run_classification('ortodoncja.csv')