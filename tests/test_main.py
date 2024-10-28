# tests/test_main.py

from main import predict_stroke

def test_predict_stroke():
    """
    Prueba la función de predicción para asegurar que devuelve un string
    y que contiene el texto esperado.
    """
    result = predict_stroke(
        gender="Male",
        age=70,
        hypertension=True,
        heart_disease=False,
        avg_glucose_level=100.5,
        smoking_status="never smoked"
    )
    assert isinstance(result, str), "El resultado de la predicción debería ser un string."
    assert "La probabilidad de ictus es" in result, "El resultado no contiene el mensaje esperado."
