#streamlit run blood_viscosity_app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import streamlit as st
from scipy.signal import find_peaks
from scipy.stats import linregress

# --- Функция модели затухающих колебаний ---
def damped_oscillation(t, h0, sigma, omega, phi):
    return h0 * np.exp(-sigma * t) * np.cos(omega * t + phi)

# --- Аппроксимация и расчет коэффициента затухания ---
def analyze_damping(t, h, sigma):
    h0_guess = np.max(h)
    sigma_guess = sigma
    omega_guess = 10
    phi_guess = 0
    popt, _ = curve_fit(damped_oscillation, t, h, p0=[h0_guess, sigma_guess, omega_guess, phi_guess])
    fit_h = damped_oscillation(t, *popt)
    sigma = popt[1]
    return popt, fit_h, sigma

# --- Расчет вязкости по затухающим колебаниям ---
def calculate_viscosity(rho, r, h0_guess, L, sigma ):
    eta = (sigma * rho * r**2 * h0_guess * (1 +  (L / (2 * h0_guess)))) / (2 * L)
    return eta

# --- Расчет вязкости по Пуазейлю ---
def poiseuille_viscosity(Q, dP, R, L):
    return (np.pi * R**4 * dP) / (8 * Q * L)

# --- Интерфейс ---
st.title("Измерение вязкости крови по методу затухающих колебаний")
st.markdown("Автор: *Лунев Никита*")

st.sidebar.header("Параметры установки")
R = st.sidebar.number_input("Радиус трубки R (м)", 0.0001, 0.04, 0.01, format="%.3f")
rho = st.sidebar.number_input("Плотность крови ρ (кг/м³)", 900, 1200, 1060)
L = st.sidebar.number_input("Длина канала (м)", value=0.1)

st.subheader("Загрузите файл с экспериментальными данными")
uploaded_file = st.file_uploader("CSV-файл (2 колонки: t, h)", type="csv")

# --- Если пользователь ничего не загрузил, подставляем симуляцию ---
if not uploaded_file:
    h0 = 0.004
    sigma = np.log(3) / 6
    omega = 1
    phi = 0
    # Время от 0 до 10 секунд
    t = np.linspace(0, 10, 100)
    h = h0 * np.exp(-sigma * t) * np.cos(omega * t + phi)
    noise = np.random.normal(0, 0.0002, size=h.shape)
    h = h + noise
    df = pd.DataFrame({'t': t, 'h': h})
else:
    df = pd.read_csv(uploaded_file)
    t = df.iloc[:, 0].values
    h = df.iloc[:, 1].values
    # 1. Нахождение индексов пиков
    peaks, _ = find_peaks(np.abs(h), distance=10)  # distance можно подогнать

    # 2. Время и значения пиков
    t_peaks = t[peaks]
    h_peaks = np.abs(h[peaks])
    ln_h = np.log(h_peaks)

    # 3. Линейная регрессия ln(h) = ln(A) - sigma * t
    slope, intercept, r_value, p_value, std_err = linregress(t_peaks, ln_h)
    sigma = -slope  # Минус, потому что наклон убывающий

st.write("Фрагмент данных:")
st.dataframe(df.head())

params, fit_h, sigma = analyze_damping(t, h,sigma)
eta = calculate_viscosity(rho, R, params[0],L, sigma)

st.success(f"Коэффициент затухания σ = {sigma:.4f} 1/с")
st.success(f"Вязкость η = {eta:.6f} Па·с")

# --- График h(t) и аппроксимации ---
st.subheader("📈 График колебаний")
fig1, ax1 = plt.subplots()
ax1.plot(t, h, 'bo', label='Эксперимент')
ax1.plot(t, fit_h, 'r-', label='Аппроксимация')
ax1.set_xlabel("Время (с)")
ax1.set_ylabel("Уровень h(t) (м)")
ax1.set_title("Затухающие колебания")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)



# --- Блок Пуазейля ---
st.subheader("Сравнение с методом Пуазейля (необязательно)")
with st.expander("Метод Пуазейля"):
    V = st.number_input("Объем крови в (мл)", min_value=0.0001, value=1.0)
    t_pua = st.number_input("Время переливания (с)", min_value=0.0, value=0.0, format="%.2f")
    dP = st.number_input("Перепад давления ΔP (Па)", min_value=0.0, value=1000.0)
    L_pua = st.number_input("Длина трубки L (м)", min_value=0.0001, value=0.2)
    r_pua = st.number_input("Радиус трубки прибора (м)", min_value=0.0001, value=0.00055, format="%.5f")

    if (V > 0 and t_pua>0 and dP > 0 and L_pua > 0 and r_pua>0):
        Q = (V*10**(-6))/t_pua
        V_modi = V*10**(-6)
        eta_poiseuille = poiseuille_viscosity(Q, dP, r_pua, L_pua)
        diff = abs(eta - eta_poiseuille) / eta * 100
        st.info(f"Обьем: V = {V_modi:.6f} m3")
        st.info(f"Пуазейль: η = {eta_poiseuille:.6f} Па·с")
        st.warning(f"Разница: {diff:.2f} %")
