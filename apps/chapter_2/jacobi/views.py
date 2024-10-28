import numpy as np
from django.shortcuts import render

def jacobi_view(request):
    context = {
        'range_matrices': range(2, 7),
        'matrix_size': None,
        'original_matrix': None,
        'b_values': None,
        'x0_values': None,
        'tol': None,
        'niter': None,
        'iteration_table': None
    }

    if request.method == 'POST':
        try:
            # Get matrix size from form
            matrix_size = int(request.POST.get('matrix_size', 3))  # Default to 3x3 matrix if not provided
            context['matrix_size'] = range(matrix_size)

            # Retrieve matrix A, vector b, and vector x0
            A = []
            b = []
            x0 = []
            for i in range(matrix_size):
                row = []
                for j in range(matrix_size):
                    value = request.POST.get(f'A_{i}_{j}', '0')
                    row.append(float(value))
                A.append(row)
                b.append(float(request.POST.get(f'b_{i}', '0')))
                x0.append(float(request.POST.get(f'x0_{i}', '0')))

            # Retrieve tolerance and iterations
            tol = float(request.POST.get('tol'))
            niter = int(request.POST.get('niter'))

            # Store original matrix and vectors for display
            context['original_matrix'] = A
            context['b_values'] = b
            context['x0_values'] = x0
            context['tol'] = tol
            context['niter'] = niter

            # Call Jacobi method
            solution, error, matrices_by_iteration, iteration_table = jacobi_method(A, b, x0, tol, niter)
            context['iteration_table'] = iteration_table

        except Exception as e:
            context['error'] = f"Error: {e}"

    return render(request, 'jacobi.html', context)


def jacobi_method(A, b, x0, tol, max_iter):
    A = np.array(A)
    b = np.array(b)
    x = np.array(x0)
    n = len(b)
    x_new = np.zeros_like(x)
    iteration_table = []

    for k in range(max_iter):
        for i in range(n):
            sum_ = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sum_) / A[i][i]

        iteration_table.append((k+1, x_new.copy(), np.linalg.norm(x_new - x)))

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new.tolist(), np.linalg.norm(x_new - x, ord=np.inf), None, iteration_table

        x = np.copy(x_new)

    return x.tolist(), np.linalg.norm(x_new - x, ord=np.inf), None, iteration_table
