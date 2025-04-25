import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('rf.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
     "aniongap_min": {"type": "numerical", "min": 0.000, "max": 100.000, "default": 8, "unit": "mmol/L"},
    "bun_max": {"type": "numerical", "min": 0, "max": 300.000, "default": 10, "unit": "mg/dL"},
    "gcs_min": {"type": "numerical", "min": 3, "max": 15, "default": 15, "unit": ""},
    "heart_rate_max": {"type": "numerical", "min": 0, "max": 1000, "default": 60, "unit": "bpm"},
    "liver_disease": {"type": "categorical", "options": [0, 1], "default": 0, "unit": "0=No, 1=Yes"},
    "magnesium_min": {"type": "numerical", "min": 0, "max": 100, "default": 2, "unit": "mg/dL"},
    "mbp_max": {"type": "numerical", "min": 0, "max": 1000, "default": 65, "unit": "mmHg"},
    "mv": {"type": "categorical", "options": [0, 1], "default": 0, "unit": "0=No, 1=Yes"},
    "ph_min": {"type": "numerical", "min": 0, "max": 14, "default": 7.4, "unit": ""},
    "postaki_creatinine_max": {"type": "numerical", "min": 0, "max": 150, "default": 1, "unit": "mg/dL"},
    "postaki_rrt_24h": {"type": "categorical", "options": [0, 1], "default": 0, "unit": "0=No, 1=Yes"},
    "resp_rate_min": {"type": "numerical", "min": 0, "max": 200, "default": 12, "unit": "breaths/min"},
    "rrt_24h": {"type": "categorical", "options": [0, 1], "default": 0, "unit": "0=No, 1=Yes"},
    "sedative_drug": {"type": "categorical", "options": [0, 1], "default": 0, "unit": "0=No, 1=Yes"},
    "sepsis": {"type": "categorical", "options": [0, 1], "default": 0, "unit": "0=No, 1=Yes"},
    "sodium_max": {"type": "numerical", "min": 0, "max": 300, "default": 140, "unit": "mmol/L"},
    "temperature_max": {"type": "numerical", "min": 0, "max": 50, "default": 37, "unit": "°C"},
    "urineoutput_24h": {"type": "numerical", "min": 0, "max": 50000, "default": 2000, "unit": "mL/24h"},
}
# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        unit = properties.get("unit", "")
        if unit:
            label = f"{feature} [{properties['min']} - {properties['max']}] ({unit})"
        else:
            label = f"{feature} [{properties['min']} - {properties['max']}]"
        step = 0.01 if properties.get("step") is None else properties["step"]
        value = st.number_input(
            label=label,
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            step=step
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (0=No, 1=Yes)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 提取类别 1 的概率
    probability_class_1 = predicted_proba[1] * 100
    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of AKI-associated delirium is {probability_class_1  :.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = 1  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
