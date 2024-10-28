from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt  # Asegúrate de importar matplotlib


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
                    value = request.POST.get(f'A_{i}_{j}', 1)  # Valor por defecto de 1
                    try:
                        row.append(float(value))  # Convertir a float
                    except ValueError:
                        row.append(1.0)  # Si falla, usar valor por defecto
                A.append(row)

            # Procesar el vector b
            for i in range(matrix_size):
                value_b = request.POST.get(f'b_{i}', 1) 
                try:
                    b.append(float(value_b)) 
                except ValueError:
                    b.append(1.0)  

            # Procesar el vector x0
            for i in range(matrix_size):
                value_x0 = request.POST.get(f'x0_{i}', 0)  
                try:
                    x0.append(float(value_x0))
                except ValueError:
                    x0.append(0.0) 

            print(f'A:\n{A}')
            print(f'b:\n{b}')
            print(f'x0:\n{x0}')

            spectral_radius = np.max(np.abs(np.linalg.eigvals(A)))


            solution, error, matrices_by_iteration, iteration_table = gauss_seidel_method(A, b, x0, tol, niter)

            graph_path = None
            if matrix_size == 2:
                graph_path = 'static/graphs/sistema.png'
                svg_path = 'static/graphs/sistema.svg'
                graph_system(A, b, graph_path, svg_path)  # Graficar el sistema

            context = {
                'matrix_size': matrix_size,
                'original_matrix': A,
                'matrix_data': {'A': A, 'b': b, 'x0': x0},
                'matrices_by_iteration': matrices_by_iteration,
                'solution': solution,
                'relative_error': error,
                'iteration_table': iteration_table,
                'table_size': len(iteration_table),
                'spectral_radius': spectral_radius,
                'range_matrices': range(2, 7),
                'graph_path': graph_path, 
                'niter': niter,
            }

        except Exception as e:
            context['error'] = str(e)
    else:
        context['matrix_size'] = max_matrix_size
        context['matrix_data'] = {'A': [], 'b': [], 'x0': []}

    context['range'] = range(2, 7) 
    
    return render(request, 'gauss_seidel.html', context)


    def gauss_seidel_method(A, b, x0, tol=1e-5, max_iter=20):

        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        x = np.array(x0, dtype=float)
        n = len(b)
        
        norm_val = np.inf
        itr = 0 

        matrices_by_iteration = []
        iteration_table = []

        while norm_val > tol and itr < max_iter:
            x_old = np.copy(x) 
            x_new = np.copy(x) 
            
            for i in range(n):
                sigma = 0
                
                for j in range(i):
                    sigma += A[i, j] * x_new[j]
                
                for j in range(i + 1, n):
                    sigma += A[i, j] * x_old[j]

                x_new[i] = (1 / A[i, i]) * (b[i] - sigma)

            norm_val = np.linalg.norm(x_new - x, ord=np.inf)  # Norma infinita
            if norm_val < tol:
                print("Iteration table: ", iteration_table)
                return x_new, norm_val, matrices_by_iteration, iteration_table
            
            x = np.copy(x_new)
            itr += 1 


            iteration_table.append((itr, x_new.copy(), norm_val)) 
            matrices_by_iteration.append(x.copy())

        return x, norm_val, matrices_by_iteration, iteration_table


def graph_system(A, b, path_png='static/graphs/sistema.png', path_svg='static/graphs/sistema.svg'):
    m1, m2 = A[0]
    m3, m4 = A[1]

    x_vals = np.linspace(-10, 10, 400)
    y1_vals = (b[0] - m1 * x_vals) / m2
    y2_vals = (b[1] - m3 * x_vals) / m4

    x_min = min(x_vals)
    x_max = max(x_vals)
    y_min = min(min(y1_vals), min(y2_vals))
    y_max = max(max(y1_vals), max(y2_vals))

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y1_vals, label='Ecuación 1', color='blue')
    plt.plot(x_vals, y2_vals, label='Ecuación 2', color='orange', linestyle='--')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title('Gráfica del Sistema de Ecuaciones')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()

    # Guardar la gráfica como PNG
    plt.savefig(path_png)

    # Guardar la gráfica como SVG
    plt.savefig(path_svg, format='svg')

    plt.close()

