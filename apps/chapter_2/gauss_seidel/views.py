from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt

def gauss_seidel(request):
    context = {
        'range_matrices': range(2, 7),
    }
    max_matrix_size = 6

    if request.method == 'POST':
        try:
            matrix_size = int(request.POST.get('matrix_size', max_matrix_size))

            A = []
            b = []
            x0 = []
            tol = float(request.POST.get('tol', 1e-5))  # Valor por defecto para tolerancia
            niter = int(request.POST.get('niter', 100))  # Valor por defecto para iteraciones

            # Procesar la matriz A
            for i in range(matrix_size):
                row = []
                for j in range(matrix_size):
                    value = request.POST.get(f'A_{i}_{j}', 0)  # Valor por defecto de 0
                    try:
                        row.append(float(value))  # Convertir a float
                    except ValueError:
                        row.append(0.0)  # Si falla, usar valor por defecto
                A.append(row)

            # Procesar el vector b
            for i in range(matrix_size):
                value_b = request.POST.get(f'b_{i}', 0)
                try:
                    b.append(float(value_b))
                except ValueError:
                    b.append(0.0)

            # Procesar el vector x0
            for i in range(matrix_size):
                value_x0 = request.POST.get(f'x0_{i}', 0)
                try:
                    x0.append(float(value_x0))
                except ValueError:
                    x0.append(0.0)

            spectral_radius = compute_spectral_radius(A)

            solution, error, matrices_by_iteration, iteration_table = gauss_seidel_method(A, b, x0, tol, niter)

            # Generar gráfica si el tamaño de la matriz es 2x2
            graph_paths = None
            if matrix_size == 2:
                graph_paths = graph_system(A, b)

            context = {
                'matrix_size': range(matrix_size),
                'original_matrix': A,
                'b_values': b,
                'x0_values': x0,
                'solution': solution,
                'relative_error': error,
                'iteration_table': iteration_table,
                'result_message': f"Convergió exitosamente en {len(iteration_table)} iteraciones.",
                'spectral_radius': spectral_radius,
                'range_matrices': range(2, 7),
                'niter': niter,
                'tol': tol,
                'graph_png': graph_paths['png'] if graph_paths else None,
                'graph_svg': graph_paths['svg'] if graph_paths else None,
            }

            # Verificar si el método no convergió
            if len(iteration_table) >= niter:
                context['error'] = f"El método no convergió en {niter} iteraciones."
                context['result_message'] = f"El método no convergió en {niter} iteraciones."

        except Exception as e:
            context['error'] = f"Error: {str(e)}"

    else:
        context['matrix_size'] = range(max_matrix_size)
        context['matrix_data'] = {'A': [], 'b': [], 'x0': []}

    return render(request, 'gauss_seidel.html', context)

def compute_spectral_radius(A):
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    
    # Matriz iterativa T
    T = -np.linalg.inv(D + L) @ U
    
    # Radio espectral de T
    spectral_radius = np.max(np.abs(np.linalg.eigvals(T)))
    return spectral_radius



def gauss_seidel_method(A, b, x0, tol=1e-5, max_iter=20):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.array(x0, dtype=float)
    n = len(b)

    norm_val = np.inf
    itr = 0

    iteration_table = [(0, x.tolist(), None)]  # Iteración inicial

    while norm_val > tol and itr < max_iter:
        x_old = np.copy(x)
        x_new = np.copy(x)

        for i in range(n):
            sigma = 0

            for j in range(n):
                if j != i:
                    sigma += A[i, j] * x_new[j]  # Usar x_new para Gauss-Seidel

            x_new[i] = (b[i] - sigma) / A[i, i]

        norm_val = np.linalg.norm(x_new - x_old, ord=np.inf)  # Norma infinita
        error = norm_val

        x = np.copy(x_new)
        itr += 1

        iteration_table.append((itr, x_new.tolist(), error))

        if norm_val < tol:
            return x_new.tolist(), error, None, iteration_table

    return x.tolist(), error, None, iteration_table


def graph_system(A, b, png_path='static/graphs/gauss_seidel_system.png', svg_path='static/graphs/gauss_seidel_system.svg'):
    """Genera y guarda la gráfica del sistema de ecuaciones lineales."""
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
