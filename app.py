import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import shap

app = Flask(__name__)

# --- загрузка модели ---
with open("model.pkl", "rb") as f:
    model = pickle.load(f)   # это XGBClassifier

feature_names = [
    "Gender", "Age", "Height", "Weight",
    "FHWO",
    "FAVC", "FCVC", "NCP", "CAEC", "SMOKE",
    "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS",
]

label_map = {
    "Gender": "Пол (0 – женщина, 1 – мужчина)",
    "Age": "Возраст (лет)",
    "Height": "Рост (м)",
    "Weight": "Вес (кг)",
    "FCVC": "Как часто вы употребляете овощи? 1 – редко, 2 – иногда, 3 – часто",
    "FAVC": "Часто ли вы употребляете высококалорийную еду? 0 – нет, 1 – да",
    "NCP": "Как много основных приёмов пищи у вас в день? (1–4)",
    "CAEC": "Как часто вы употребляете еду между основными приёмами пищи? 0 – никогда, 1 – редко, 2 – иногда, 3 – часто",
    "SMOKE": "Вы курите? 0 – нет, 1 – да",
    "CH2O": "Как много воды вы употребляете ежедневно? 1 – мало, 2 – среднее количество, 3 – много",
    "SCC": "Вы ведёте подсчёт калорий? 0 – нет, 1 – да",
    "FAF": "Как часто вы занимаетесь физической активностью? 0 – никогда, 1 – редко, 2 – иногда, 3 – часто",
    "TUE": "Как часто вы занимаетесь малоподвижной деятельностью? 0 – редко, 1 – иногда, 2 – часто",
    "CALC": "Как часто вы употребляете алкоголь? 0 – никогда, 1 – редко, 2 – иногда, 3 – часто",
    "MTRANS": "Каким видом транспорта вы пользуетесь чаще всего? 0 – автомобиль, 1 – мотоцикл, 2 – общественный транспорт, 3 – велосипед, 4 – пешком",
    "FHWO": "Есть ли у родственников проблемы с ожирением? 0 – нет, 1 – да",
}
#'Automobile': 0, 'Motorbike': 1, 'Public_Transportation': 2, 'Bike': 3, 'Walking': 4

# SHAP-эксплейнер для XGBoost
explainer = shap.TreeExplainer(model)

classes = [
    "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
    "Overweight_Level_II", "Obesity_Type_I",
    "Obesity_Type_II", "Obesity_Type_III",
]

modifiable_features = [
    "Weight", "FCVC", "FAVC", "NCP", "CAEC", "CH2O",
    "FAF", "TUE", "CALC", "MTRANS", "SMOKE", "SCC", "FHWO",
]

def make_recommendations(x_row, shap_row, class_id):
    recs = []

    is_underweight = (class_id == 0)
    is_normal      = (class_id == 1)
    is_overweight  = (class_id >= 2)

    def add_if_obesity(name, text, value_condition=None):
        """Рекомендации для избыточного веса / ожирения:
        признак повышает риск (SHAP > 0) → совет уменьшить фактор."""
        if is_overweight and name in feature_names:
            j = feature_names.index(name)
            shap_val = shap_row[j]
            val = x_row[name]  

            if shap_val > 0:           # признак увеличивает риск этого класса
                if value_condition is None or value_condition(val):
                    recs.append(text)
            #if shap_row[j] > 0:
             #   recs.append(text)

    # --- для избыточного веса / ожирения ---
    add_if_obesity(
        "Weight",
        "Высокий показатель веса является основным признаком ожирения. "
        "Вам стоит снизить вес."
    )
    add_if_obesity(
        "FAVC",
        "Частое употребление высококалорийной пищи увеличивает прогнозируемый риск ожирения. "
        "Попробуйте сократить фастфуд и сладости.",
        value_condition=lambda v: v >= 1
    )
    add_if_obesity(
        "FCVC",
        "Низкая частота употребления овощей связана с более высоким риском ожирения. "
        "Повышение доли овощей в рационе может снизить риск.",
        value_condition=lambda v: v <= 2
    )
    add_if_obesity(
        "CH2O",
        "Недостаточное потребление воды связано с повышенным риском ожирения. "
        "Увеличьте объём потребляемой воды при отсутствии противопоказаний.",
        value_condition=lambda v: v <= 2
    )
    add_if_obesity(
        "CALC",
        "Частое употребление калорийных напитков/алкоголя ухудшает прогноз по ожирению. "
        "Сокращение таких напитков может снизить риск.",
        value_condition=lambda v: v >= 2
    )
    add_if_obesity(
        "CAEC",
        "Злоупотребление перекусами между основными приемами пищи может повышать риск ожирения. "
        "Сокращение перекусов улучшит прогноз.",
        value_condition=lambda v: v >= 2
    )

    if is_underweight or is_normal:
        return []

    if not recs:
        recs.append(
            "Модель не выявила ярко выраженных факторов, значительно влияющих на ваш риск. "
        )
    return recs[:4]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    proba = None
    recs = None

    if request.method == "POST":
        data = {}
        cat_features = ["Gender", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "FHWO"]

        for name in feature_names:
            val = request.form.get(name)
            if val is None or val == "":
                val = 0

            if name in cat_features:
                data[name] = int(val)   # закодированные категории 0..N
            else:
                data[name] = float(val)

        X_user = pd.DataFrame([data], columns=feature_names)

        # --- предсказание ---
        y_proba = model.predict_proba(X_user)[0]
        class_id = int(np.argmax(y_proba))
        prediction = classes[class_id]
        proba = list(zip(classes, y_proba.round(3)))

        # --- SHAP для одного пользователя ---
        shap_vals = explainer.shap_values(X_user)   # (1, n_features, n_classes)
        shap_row = shap_vals[0, :, class_id]        # (n_features,)

        user_importance = np.abs(shap_row)
        top_idx = np.argsort(user_importance)[::-1][:5]
        top_features = [(feature_names[i], shap_row[i]) for i in top_idx]

        recs = make_recommendations(X_user.iloc[0], shap_row, class_id)

    return render_template(
        "index.html",
        feature_names=feature_names,
        label_map=label_map,
        prediction=prediction,
        proba=proba,
        recs=recs,
    )

if __name__ == "__main__":
    app.run(debug=True)

