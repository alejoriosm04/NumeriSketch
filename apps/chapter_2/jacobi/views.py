import numpy as np
from django.shortcuts import render

def jacobi_view(request):
    context = {
        'range_matrices': range(2, 7), 
    }
    max_matrix_size = 6  

    if request.method == 'POST':
        try:
            # Get the matrix size selected by the user
            matrix_size = int(request.POST.get('matrix_size', max_matrix_size))

            A = []
            b = []
            x0 = []
            tol = float(request.POST.get('tol'))
            niter = int(request.POST.get('niter'))

            # Process matrix A (up to matrix_size x matrix_size)
            for i in range(matrix_size):
                row = []
                for j in range(matrix_size):
                    value = request.POST.get(f'A_{i}_{j}')
                    if value:
                        row.append(float(value))
                if row:
                    A.append(row)

            # Process vector b
            for i in range(matrix_size):
                value_b = request.POST.get(f'b_{i}')
                if value_b:
                    b.append(float(value_b))

            # Process vector x0 (initial approximation)
            for i in range(matrix_size):
                value_x0 = request.POST.get(f'x0_{i}')
                if value_x0:
                    x0.append(float(value_x0))

            # Execute Jacobi method
            solution, error, matrices_by_iteration, iteration_table = jacobi_method(A, b, x0, tol, niter)

            context = {
                'matrix_size': matrix_size,
                'matrix_data': {'A': A, 'b': b, 'x0': x0},
                'original_matrix': A,
                'matrices_by_iteration': matrices_by_iteration,
                'solution': solution,
                'relative_error': error,
                'iteration_table': iteration_table,
            }

        except Exception as e:
            context['error'] = str(e)

    else:
        context['matrix_size'] = max_matrix_size
        context['matrix_data'] = {'A': [], 'b': [], 'x0': []}

    context['range'] = range(2, 7)  # Generates the options 2x2 to 6x6

    return render(request, 'jacobi.html', context)

def jacobi_method(A, b, x0, tol, max_iter):
    A = np.array(A)
    b = np.array(b)
    x = np.array(x0)
    n = len(b)
    x_new = np.zeros_like(x)
    matrices_by_iteration = {}
    iteration_table = []

    for k in range(max_iter):
        for i in range(n):
            sum_ = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_) / A[i][i]

        matrices_by_iteration[k+1] = x_new.copy()

        iteration_table.append((k+1, x_new.copy(), np.linalg.norm(x_new - x)))

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, np.linalg.norm(x_new - x, ord=np.inf), matrices_by_iteration, iteration_table

        x = np.copy(x_new)

    return x, np.linalg.norm(x_new - x, ord=np.inf), matrices_by_iteration, iteration_table
