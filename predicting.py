import pandas as pd
import joblib


def predict_from_file(input_file, output_file):
    try:
        # 1. Wczytanie modelu i pomocniczych plików
        model = joblib.load('model_orto.joblib')
        le = joblib.load('encoder_klas.joblib')
        expected_features = joblib.load('lista_cech.joblib')

        # 2. Wczytanie nowych danych
        new_data = pd.read_csv(input_file)

        # Upewnienie się, że mamy wszystkie potrzebne kolumny
        X_new = new_data[expected_features]

        # 3. Predykcja
        predictions_num = model.predict(X_new)
        predictions_labels = le.inverse_transform(predictions_num)

        # 4. Zapisanie wyników
        new_data['PREDICTED_growth_direction'] = predictions_labels
        new_data.to_csv(output_file, index=False)

        print(f"Predykcja zakończona. Wyniki zapisano w: {output_file}")

    except FileNotFoundError:
        print("Błąd: Nie znaleziono pliku modelu lub danych wejściowych.")
    except Exception as e:
        print(f"Wystąpił błąd: {e}")


if __name__ == "__main__":
    # Przykład użycia:
    # predict_from_file('nowi_pacjenci.csv', 'wyniki_pacjentow.csv')
    print("Skrypt gotowy. Użyj funkcji predict_from_file(input, output).")