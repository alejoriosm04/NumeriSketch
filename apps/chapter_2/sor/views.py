import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render

# Modificación en sor_view para mostrar advertencias solo si no converge
def sor_view(request):
    context = {
        'range_matrices': range(2, 7),
        'matrix_size': None,
        'original_matrix': None,
        'b_values': None,
        'x0_values': None,
        'tol': None,
        'niter': None,
        'iteration_table': None,
        'result_message': None,
        'warning_message': None,
        'spectral_radius': None,  
        'graph_png': None,
        'graph_svg': None
    }

    if request.method == 'POST':
        try:
            matrix_size = int(request.POST.get('matrix_size', 3))
            context['matrix_size'] = range(matrix_size)

            A, b, x0 = [], [], []
            for i in range(matrix_size):
                row = [float(request.POST.get(f'A_{i}_{j}', '0')) for j in range(matrix_size)]
                A.append(row)
                b.append(float(request.POST.get(f'b_{i}', '0')))
                x0.append(float(request.POST.get(f'x0_{i}', '0')))

            tol = float(request.POST.get('tol'))
            niter = int(request.POST.get('niter'))
            w = float(request.POST.get('w', 1.0))  # Factor de relajación

            context.update({
                'original_matrix': A,
                'b_values': b,
                'x0_values': x0,
                'tol': tol,
                'niter': niter,
                'w': w 
            })

            # Llamada al método de SoR
            solution, error, message, iteration_table, warning, spectral_radius = sor_method(A, b, x0, tol, niter, w)
            context['iteration_table'] = iteration_table
            context['result_message'] = message
            context['spectral_radius'] = spectral_radius

            if spectral_radius < 1:
                context['convergence_message'] = "El radio espectral en el método SOR indica la rapidez de convergencia. Para que el método sea eficiente, este valor debe ser menor que 1, ya que garantiza la disminución progresiva del error en cada iteración. Ajustar el parámetro de relajación ω permite minimizar el radio espectral y optimizar la velocidad de convergencia."
            else:
                context['convergence_message'] = "El método SoR no convergerá ya que el radio espectral es mayor o igual a 1."

            if "Fracasó" in message and warning:
                context['warning_message'] = "Advertencia: La matriz puede no ser adecuada para la convergencia."

            if matrix_size == 2:
                graph_paths = graph_system(A, b)
                context.update({
                    'graph_png': graph_paths['png'],
                    'graph_svg': graph_paths['svg']
                })

        except ValueError as ve:
            context['result_message'] = f"Error de validación: {ve}"
        except Exception as e:
            context['result_message'] = f"Error inesperado: {e}"

    return render(request, 'sor.html', context)


def sor_method(A, b, x0, tol, max_iter, w):
    A = np.array(A)
    b = np.array(b)
    x = np.array(x0)
    n = len(b)
    iteration_table = [(0, x.tolist(), None)]  # Iteración inicial
    warning = False

    # Verificación de diagonal dominante
    is_diagonally_dominant = all(
        abs(A[i][i]) > sum(abs(A[i][j]) for j in range(n) if j != i) for i in range(n)
    )
    if not is_diagonally_dominant:
        warning = True

    # Calcular la matriz de iteración T para SoR
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    
    # T es la matriz de iteración para SoR
    T = np.linalg.inv(D + w * L) @ ((1 - w) * D - w * U)

    # Calcular el radio espectral (máximo valor absoluto de los eigenvalores de T)
    spectral_radius = max(abs(np.linalg.eigvals(T)))

    # Iteraciones de SoR
    for k in range(1, max_iter + 1):
        x_old = np.copy(x)
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (1 - w) * x_old[i] + (w / A[i][i]) * (b[i] - sigma)

        error = np.linalg.norm(x - x_old, ord=np.inf)
        iteration_table.append((k, x.tolist(), error))

        if error < tol:
            return x.tolist(), error, f"Convergió exitosamente en la iteración {k}.", iteration_table, warning, spectral_radius

    return x.tolist(), error, f"Fracasó en {max_iter} iteraciones.", iteration_table, warning, spectral_radius


def graph_system(A, b, png_path='static/graphs/sor_system.png', svg_path='static/graphs/sor_system.svg'):
    m1, m2 = A[0]
    m3, m4 = A[1]

    x_vals = np.linspace(-10, 10, 400)
    y1_vals = (b[0] - m1 * x_vals) / m2
    y2_vals = (b[1] - m3 * x_vals) / m4

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y1_vals, label='Ecuación 1', color='blue')
    plt.plot(x_vals, y2_vals, label='Ecuación 2', color='orange', linestyle='--')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title('Gráfica del Sistema de Ecuaciones')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()

    plt.savefig(png_path, format='png')
    plt.savefig(svg_path, format='svg')
    plt.close()

    return {'png': png_path, 'svg': svg_path}