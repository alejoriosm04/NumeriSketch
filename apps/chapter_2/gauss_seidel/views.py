from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import json
from django.shortcuts import render

@csrf_exempt
def gauss_seidel(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print("Received Data:", data)  # Log the incoming data for debugging

            # Validate the incoming data
            matrix = np.array(data.get('matrix', []))
            vectorB = np.array(data.get('vectorB', []))

            # Change this line to access initialGuess instead of initial_guess
            if isinstance(data['initialGuess'], list):
                initial_guess = np.array(data['initialGuess'], dtype=float)
            else:
                # If scalar, replicate it to match the size of vectorB
                initial_guess = np.full_like(vectorB, fill_value=float(data['initialGuess']))

            tolerance = float(data.get('tolerance', 1e-5))
            max_iterations = int(data.get('maxIterations', 10))  # Corrected the key as well

            n = len(vectorB)
            x = np.copy(initial_guess)
            prev_x = np.zeros_like(x)
            iterations = []

            # Decompose matrix A into lower and upper triangular components
            L = np.tril(matrix)  # Lower triangular component
            U = np.triu(matrix, k=1)  # Upper triangular component

            # Calculate L^-1 (inverse of lower triangular matrix)
            L_inv = np.linalg.inv(L)

            # Calculate T and C for the system Lx = b
            C = np.dot(L_inv, vectorB)

            # Gauss-Seidel Iteration using L and C
            for iteration in range(max_iterations):
                prev_x = np.copy(x)
                for i in range(n):
                    sum1 = np.dot(L[i, :i], x[:i])
                    x[i] = (C[i] - sum1) / L[i, i]  # Update x using L and C

                # Calculate errors
                abs_error = np.linalg.norm(x - prev_x, ord=np.inf)  # Absolute error
                rel_error = abs_error / (np.linalg.norm(x, ord=np.inf) + 1e-10)  # Relative error to avoid division by 0

                iterations.append({
                    'iteration': iteration + 1,
                    'x_values': x.tolist(),
                    'absolute_error': abs_error,
                    'relative_error': rel_error
                })

                # Check if the error is less than the tolerance
                if abs_error < tolerance:
                    break

            # Calculate spectral radius
            eigvals = np.linalg.eigvals(matrix)
            spectral_radius = max(abs(eigvals))

            return JsonResponse({
                'iterations': iterations,
                'spectral_radius': spectral_radius
            })

        except Exception as e:
            print("Error:", e)  # Log the error
            return JsonResponse({'error': str(e)}, status=400)
    
    return render(request, 'gauss_seidel.html')
