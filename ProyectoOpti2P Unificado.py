# 1. CARGA Y PREPARACIÓN DE DATOS
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop(columns=['Wear_Class', 'VB_mm']).values
    y = df['VB_mm'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


filepath = os.path.expanduser("~/Downloads/tool_wear_dataset.csv")
X_train_base, X_test_base, y_train, y_test = load_and_prepare_data(filepath)
n_features = X_train_base.shape[1]
print("Bloque 1: Datos cargados y estandarizados correctamente.")

# 2. FUNCIONES DEL MODELO LINEAL
# ==========================================
def objective_function_lin(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)


def gradient_lin(X, y, theta):
    m = len(y)
    return (1 / m) * X.T.dot(X.dot(theta) - y)


def hessian_lin(X):
    m = X.shape[0]
    return (1 / m) * X.T.dot(X)

# 3. ALGORITMOS PARA EL MODELO LINEAL
# ==========================================
def newton_method_linear(X, y, theta_init, tol=1e-6, max_iter=100):
    theta = theta_init.copy()
    J_history = []
    H = hessian_lin(X)
    print(f"\n[Newton Lineal] Condición de Hessiana: {np.linalg.cond(H):.2f}")

    for i in range(max_iter):
        J_history.append(objective_function_lin(X, y, theta))
        grad = gradient_lin(X, y, theta)

        if np.linalg.norm(grad) < tol:
            print(f"[Newton Lineal] Convergencia en iteración {i}.")
            break

        s_k = np.linalg.solve(H, -grad)
        theta = theta + s_k

    return theta, J_history


def trust_region_linear(X, y, theta_init, tol=1e-6, max_iter=100, alpha_init=1.0):
    theta = theta_init.copy()
    J_history = []
    H = hessian_lin(X)
    alpha = alpha_init
    I = np.eye(len(theta))

    for i in range(max_iter):
        J_current = objective_function_lin(X, y, theta)
        J_history.append(J_current)
        grad = gradient_lin(X, y, theta)

        if np.linalg.norm(grad) < tol:
            print(f"[Trust Region Lineal] Convergencia en iteración {i}.")
            break

        step_accepted = False
        while not step_accepted:
            H_mod = H + alpha * I
            try:
                s_k = np.linalg.solve(H_mod, -grad)
            except np.linalg.LinAlgError:
                alpha *= 10
                continue

            theta_new = theta + s_k
            J_new = objective_function_lin(X, y, theta_new)

            actual_reduction = J_current - J_new
            predicted_reduction = - (grad.T.dot(s_k) + 0.5 * s_k.T.dot(H).dot(s_k))[0, 0]
            rho = actual_reduction / predicted_reduction if predicted_reduction != 0 else 0

            if rho > 0:
                step_accepted = True
                theta = theta_new
                if rho > 0.75:
                    alpha = max(1e-5, alpha / 3)
                elif rho < 0.25:
                    alpha *= 2
            else:
                alpha *= 10

    return theta, J_history

# 4. FUNCIONES DEL MODELO NO LINEAL
# ==========================================
def nonlinear_model(X, theta):
    n = X.shape[1]
    theta_0 = theta[0, 0]
    theta_linear = theta[1:n + 1].reshape(1, -1)
    theta_nonlinear = theta[n + 1:2 * n + 1].reshape(1, -1)
    return theta_0 + np.sum(theta_linear * np.tanh(X * theta_nonlinear), axis=1, keepdims=True)


def objective_function_nl(X, y, theta):
    m = len(y)
    predictions = nonlinear_model(X, theta)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)


def compute_gradient_fd(X, y, theta, epsilon=1e-5):
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        t_plus, t_minus = theta.copy(), theta.copy()
        t_plus[i, 0] += epsilon
        t_minus[i, 0] -= epsilon
        grad[i, 0] = (objective_function_nl(X, y, t_plus) - objective_function_nl(X, y, t_minus)) / (2 * epsilon)
    return grad


def compute_hessian_fd(X, y, theta, epsilon=1e-4):
    num_params = len(theta)
    H = np.zeros((num_params, num_params))
    for i in range(num_params):
        t_plus, t_minus = theta.copy(), theta.copy()
        t_plus[i, 0] += epsilon
        t_minus[i, 0] -= epsilon
        g_plus = compute_gradient_fd(X, y, t_plus, epsilon)
        g_minus = compute_gradient_fd(X, y, t_minus, epsilon)
        H[:, i] = ((g_plus - g_minus) / (2 * epsilon)).flatten()
    return (H + H.T) / 2

# 5. ALGORITMOS PARA EL MODELO NO LINEAL
# ==========================================
def newton_method_nl(X, y, theta_init, tol=1e-4, max_iter=50):
    theta = theta_init.copy()
    J_history = []

    for i in range(max_iter):
        J_history.append(objective_function_nl(X, y, theta))
        grad = compute_gradient_fd(X, y, theta)

        if np.linalg.norm(grad) < tol:
            print(f"[Newton No Lineal] Convergencia en iteración {i}.")
            break

        H = compute_hessian_fd(X, y, theta)
        try:
            s_k = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            print(f"[Newton No Lineal] Error: Hessiana singular/indefinida en iteración {i}. Falla el método.")
            break
        theta = theta + s_k
    return theta, J_history


def trust_region_nl(X, y, theta_init, tol=1e-4, max_iter=50, alpha_init=1.0):
    theta = theta_init.copy()
    J_history = []
    alpha = alpha_init
    I = np.eye(len(theta))

    for i in range(max_iter):
        J_current = objective_function_nl(X, y, theta)
        J_history.append(J_current)
        grad = compute_gradient_fd(X, y, theta)

        if np.linalg.norm(grad) < tol:
            print(f"[Trust Region No Lineal] Convergencia en iteración {i}.")
            break

        H = compute_hessian_fd(X, y, theta)
        step_accepted = False
        while not step_accepted:
            H_mod = H + alpha * I
            try:
                s_k = np.linalg.solve(H_mod, -grad)
            except np.linalg.LinAlgError:
                alpha *= 10
                continue

            theta_new = theta + s_k
            J_new = objective_function_nl(X, y, theta_new)

            actual_reduction = J_current - J_new
            predicted_reduction = - (grad.T.dot(s_k) + 0.5 * s_k.T.dot(H).dot(s_k))[0, 0]
            rho = actual_reduction / predicted_reduction if predicted_reduction != 0 else 0

            if rho > 0:
                step_accepted = True
                theta = theta_new
                if rho > 0.75:
                    alpha = max(1e-5, alpha / 3)
                elif rho < 0.25:
                    alpha *= 2
            else:
                alpha *= 10

    return theta, J_history

# 6. EJECUCIÓN, REPORTE Y GRÁFICAS
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("INICIANDO ENTRENAMIENTO DE LOS 4 ESCENARIOS")
    print("=" * 50)

    # 6.1 Preparación de Parámetros
    X_train_lin = np.c_[np.ones((X_train_base.shape[0], 1)), X_train_base]
    X_test_lin = np.c_[np.ones((X_test_base.shape[0], 1)), X_test_base]
    theta_init_lin = np.zeros((n_features + 1, 1))

    num_params_nl = 1 + 2 * n_features
    np.random.seed(42)
    theta_init_nl = np.random.randn(num_params_nl, 1) * 0.1

    # 6.2 Ejecución
    th_lin_newton, J_lin_newton = newton_method_linear(X_train_lin, y_train, theta_init_lin)
    th_lin_tr, J_lin_tr = trust_region_linear(X_train_lin, y_train, theta_init_lin)
    th_nl_newton, J_nl_newton = newton_method_nl(X_train_base, y_train, theta_init_nl)
    th_nl_tr, J_nl_tr = trust_region_nl(X_train_base, y_train, theta_init_nl)


    # 6.3 Función Auxiliar para Reportar
    def calcular_metricas(X_t, y_t, X_ts, y_ts, theta, is_linear):
        if is_linear:
            pred_t, pred_ts = X_t.dot(theta), X_ts.dot(theta)
        else:
            pred_t, pred_ts = nonlinear_model(X_t, theta), nonlinear_model(X_ts, theta)

        return {
            "MSE_Train": mean_squared_error(y_t, pred_t),
            "MSE_Test": mean_squared_error(y_ts, pred_ts),
            "R2_Train": r2_score(y_t, pred_t),
            "R2_Test": r2_score(y_ts, pred_ts)
        }


    resultados = {
        "Newton Lineal": calcular_metricas(X_train_lin, y_train, X_test_lin, y_test, th_lin_newton, True),
        "Trust Region Lineal": calcular_metricas(X_train_lin, y_train, X_test_lin, y_test, th_lin_tr, True),
        "Newton No Lineal": calcular_metricas(X_train_base, y_train, X_test_base, y_test, th_nl_newton, False),
        "Trust Region No Lineal": calcular_metricas(X_train_base, y_train, X_test_base, y_test, th_nl_tr, False)
    }

    print("\n" + "=" * 50)
    print("REPORTE COMPARATIVO FINAL")
    print("=" * 50)
    for modelo, mets in resultados.items():
        print(f"\n{modelo.upper()}:")
        print(f"  Entrenamiento -> MSE: {mets['MSE_Train']:.6f} | R^2: {mets['R2_Train']:.6f}")
        print(f"  Prueba        -> MSE: {mets['MSE_Test']:.6f} | R^2: {mets['R2_Test']:.6f}")

    # 6.4 Gráficas Comparativas 2x2
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Convergencia: Newton vs Trust Region (Modelos Separados)", fontsize=16)

    # Gráfico 1: Newton Lineal (Azul)
    axs[0, 0].plot(range(len(J_lin_newton)), J_lin_newton, marker='o', color='tab:blue', linestyle='-',
                   label='Newton Lineal')
    axs[0, 0].set_title("Modelo Lineal")
    axs[0, 0].set_xlabel("Iteraciones")
    axs[0, 0].set_ylabel("Función de Costo J(θ)")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Gráfico 2: Trust Region Lineal (Verde)
    axs[0, 1].plot(range(len(J_lin_tr)), J_lin_tr, marker='s', color='tab:green', linestyle='--',
                   label='Trust Region Lineal')
    axs[0, 1].set_title("Modelo Lineal")
    axs[0, 1].set_xlabel("Iteraciones")
    axs[0, 1].set_ylabel("Función de Costo J(θ)")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Gráfico 3: Newton No Lineal (Rojo)
    axs[1, 0].plot(range(len(J_nl_newton)), J_nl_newton, marker='o', color='tab:red', linestyle='-',
                   label='Newton No Lineal')
    axs[1, 0].set_title("Modelo No Lineal")
    axs[1, 0].set_xlabel("Iteraciones")
    axs[1, 0].set_ylabel("Función de Costo J(θ)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Gráfico 4: Trust Region No Lineal (Morado)
    axs[1, 1].plot(range(len(J_nl_tr)), J_nl_tr, marker='s', color='tab:purple', linestyle='--',
                   label='Trust Region No Lineal')
    axs[1, 1].set_title("Modelo No Lineal")
    axs[1, 1].set_xlabel("Iteraciones")
    axs[1, 1].set_ylabel("Función de Costo J(θ)")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()