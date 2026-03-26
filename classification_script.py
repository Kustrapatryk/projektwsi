import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Imb-learn do SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def main():
    # ---------------------------------------------------------
    # 0. Wczytanie danych
    # ---------------------------------------------------------
    file_path = 'ortodoncja.csv'
    try:
        df = pd.read_csv(file_path)
        print("Dane wczytane pomyślnie.\n")
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku '{file_path}'.")
        return

    # ---------------------------------------------------------
    # 1. Preprocessing i inżynieria cech
    # ---------------------------------------------------------
    print("Rozpoczynam preprocessing danych...")

    cols_9 = [col for col in df.columns if col.startswith('9_')]
    for col_9 in cols_9:
        base_name = col_9[2:]
        col_12 = f'12_{base_name}'
        if col_12 in df.columns:
            df[f'diff_{base_name}'] = df[col_12] - df[col_9]

    X = df.drop(columns=['growth direction'])
    y = df['growth direction']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    with open('raport_porownawczy_optymalny.txt', 'w', encoding='utf-8') as report_file:
        report_file.write("RAPORT Z KLASYFIKACJI - ZOPTYMALIZOWANY (F1-Macro & RobustScaler)\n")
        report_file.write("=" * 70 + "\n\n")

        # =========================================================
        # ETAP A: MODELOWANIE BEZ SMOTE
        # =========================================================
        print("\n--- ETAP 1: Trenowanie modeli BEZ SMOTE (Optymalizacja F1-Macro) ---")
        report_file.write("CZĘŚĆ 1: WYNIKI BEZ ZASTOSOWANIA SMOTE\n")
        report_file.write("-" * 40 + "\n")

        models_standard = {
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=2000, random_state=42),
                'params': {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'newton-cg']}
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
            },
            'SVC': {
                'model': SVC(random_state=42),
                'params': [
                    {'kernel': ['linear'], 'C': [0.1, 1, 10]},
                    {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1, 1]}
                ]
            }
        }

        for name, config in models_standard.items():
            print(f"Trenowanie: {name}...")
            clf = GridSearchCV(config['model'], config['params'], cv=cv, scoring='f1_macro', n_jobs=-1)
            clf.fit(X_train_scaled, y_train)

            best_model = clf.best_estimator_
            y_pred = best_model.predict(X_test_scaled)

            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title(f'{name} - BEZ SMOTE')
            plt.ylabel('Rzeczywista etykieta')
            plt.xlabel('Przewidziana etykieta')
            file_name = f'cm_bez_smote_{name.replace(" ", "_")}.png'
            plt.savefig(file_name, bbox_inches='tight')
            plt.close()

            acc_test = accuracy_score(y_test, y_pred)
            f1_mac = f1_score(y_test, y_pred, average='macro')

            report_file.write(f"\nModel: {name}\n")
            report_file.write(f"Najlepsze parametry (wg F1-Macro): {clf.best_params_}\n")
            report_file.write(f"Accuracy: {acc_test:.4f} | F1-Macro: {f1_mac:.4f}\n")
            report_file.write("Raport klasyfikacji:\n")
            report_file.write(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
            report_file.write("\n")

        # =========================================================
        # ETAP B: MODELOWANIE ZE SMOTE
        # =========================================================
        print("\n--- ETAP 2: Trenowanie modeli ZE SMOTE (Optymalizacja F1-Macro) ---")
        report_file.write("\n" + "=" * 70 + "\n")
        report_file.write("CZĘŚĆ 2: WYNIKI Z ZASTOSOWANIEM OVERSAMPLINGU (SMOTE)\n")
        report_file.write("-" * 40 + "\n")

        smote = SMOTE(random_state=42, k_neighbors=3)

        models_smote = {
            'Logistic Regression': {
                'model': LogisticRegression(max_iter=2000, random_state=42),
                'params': {'classifier__C': [0.01, 0.1, 1, 10], 'classifier__solver': ['lbfgs', 'newton-cg']}
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'classifier__n_estimators': [50, 100, 200], 'classifier__max_depth': [None, 5, 10]}
            },
            'SVC': {
                'model': SVC(random_state=42),
                'params': [
                    {'classifier__kernel': ['linear'], 'classifier__C': [0.1, 1, 10]},
                    {'classifier__kernel': ['rbf'], 'classifier__C': [0.1, 1, 10],
                     'classifier__gamma': ['scale', 'auto', 0.1, 1]}
                ]
            }
        }

        for name, config in models_smote.items():
            print(f"Trenowanie: {name} (Ze SMOTE)...")

            pipeline = ImbPipeline([
                ('smote', smote),
                ('classifier', config['model'])
            ])

            clf_smote = GridSearchCV(pipeline, config['params'], cv=cv, scoring='f1_macro', n_jobs=-1)
            clf_smote.fit(X_train_scaled, y_train)

            best_model_smote = clf_smote.best_estimator_
            y_pred_smote = best_model_smote.predict(X_test_scaled)

            cm_smote = confusion_matrix(y_test, y_pred_smote)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Oranges', xticklabels=le.classes_, yticklabels=le.classes_)
            plt.title(f'{name} - ZE SMOTE')
            plt.ylabel('Rzeczywista etykieta')
            plt.xlabel('Przewidziana etykieta')
            file_name_smote = f'cm_ze_smote_{name.replace(" ", "_")}.png'
            plt.savefig(file_name_smote, bbox_inches='tight')
            plt.close()

            acc_test_smote = accuracy_score(y_test, y_pred_smote)
            f1_mac_smote = f1_score(y_test, y_pred_smote, average='macro')

            report_file.write(f"\nModel: {name}\n")
            report_file.write(f"Najlepsze parametry (wg F1-Macro): {clf_smote.best_params_}\n")
            report_file.write(f"Accuracy: {acc_test_smote:.4f} | F1-Macro: {f1_mac_smote:.4f}\n")
            report_file.write("Raport klasyfikacji:\n")
            report_file.write(classification_report(y_test, y_pred_smote, target_names=le.classes_, zero_division=0))
            report_file.write("\n")

    print("\nZakończono! Zoptymalizowany raport został zapisany w 'raport_porownawczy_optymalny.txt'.")


if __name__ == "__main__":
    main()