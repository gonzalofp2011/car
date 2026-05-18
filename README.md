# PA2 - Car Evaluation con Machine Learning y Streamlit

## Dataset
Se utiliza el dataset Car Evaluation de UCI Machine Learning Repository.

Código usado en Colab:

```python
pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

car_evaluation = fetch_ucirepo(id=19)

X = car_evaluation.data.features
y = car_evaluation.data.targets
```

## Objetivo
Predecir la evaluación de un auto según sus características.

## Variables
- buying
- maint
- doors
- persons
- lug_boot
- safety

## Modelos
- Árbol de Decisión
- Random Forest

## Archivos para GitHub
- app.py
- requirements.txt
- anotaciones.txt
- modelos/modelo_arbol_decision.pkl
- modelos/modelo_random_forest.pkl

## Métricas referenciales del paquete generado
{
  "Árbol de Decisión": {
    "accuracy": 0.9133,
    "precision_macro": 0.6266,
    "recall_macro": 0.713,
    "f1_macro": 0.662,
    "cross_val_accuracy_mean": 0.82
  },
  "Random Forest": {
    "accuracy": 0.9422,
    "precision_macro": 0.9623,
    "recall_macro": 0.97,
    "f1_macro": 0.9658,
    "cross_val_accuracy_mean": 0.9132
  }
}