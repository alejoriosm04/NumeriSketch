import os
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import HttpResponseRedirect

def vandermonde(request):
    context = {
        'x_values': '',
        'y_values': '',
        'polynomial': None,
        'string_polynomial': None,
        'error': None,
        'graph': None,
        'input': None,
        'input_size': None
    }

    if request.method == 'POST':
        try:
            x_values = list(map(float, request.POST.get('x_values').split(',')))
            y_values = list(map(float, request.POST.get('y_values').split(',')))

            if len(x_values) != len(y_values):
                raise ValueError("Las listas de x e y deben tener la misma longitud.")
            if len(x_values) > 8:
                raise ValueError("No se permiten más de 8 datos.")

            input_size = len(x_values)
            paired_values = list(zip(x_values, y_values))

            polynomial, coefficients = vandermonde_method(x_values, y_values)
            
            formatted_polynomial = format_polynomial(coefficients)
            context['polynomial'] = formatted_polynomial
            context['string_polynomial'] = " + ".join(formatted_polynomial)
            context['input'] = True

            context['paired_values'] = paired_values
            graph_path = plot_polynomial(polynomial, x_values, y_values)
            context['graph'] = graph_path

        except Exception as e:
            context['error'] = f"Error: {str(e)}"

    return render(request, 'vandermonde.html', context)


def plot_polynomial(polynomial, x_values, y_values):

    x_plot = np.linspace(min(x_values), max(x_values), 500)
    y_plot = polynomial(x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot, label="Interpolación de Vandermonde")
    plt.scatter(x_values, y_values, color="red", label="Puntos")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    svg_path = os.path.join("static", "graphs", "vandermonde_interpolation.svg")
    plt.savefig(svg_path, format="svg")

    png_path = os.path.join("static", "graphs", "vandermonde_interpolation.png")
    plt.savefig(png_path, format="png")

    plt.close()
    
    return svg_path

def vandermonde_method(x_values, y_values):
    n = len(x_values)
    
    # Construct the Vandermonde matrix
    V = [[x**(n-i-1) for i in range(n)] for x in x_values]
    
    # Solve the system V * coefficients = y_values
    coefficients = gauss_solve(V, y_values)
    
    # Define the polynomial as a function
    def polynomial(x):
        return sum(c * (x**i) for i, c in enumerate(reversed(coefficients)))
    
    return polynomial, coefficients


"""Solve a linear system Ax = b using Gaussian elimination."""
def gauss_solve(matrix, values):
    n = len(matrix)
    
    # Augment the matrix with the values vector
    augmented_matrix = [row + [val] for row, val in zip(matrix, values)]
    
    # Forward elimination
    for i in range(n):
        # Find the pivot row and swap
        pivot_row = max(range(i, n), key=lambda r: abs(augmented_matrix[r][i]))
        augmented_matrix[i], augmented_matrix[pivot_row] = augmented_matrix[pivot_row], augmented_matrix[i]
        
        # Make sure the pivot is not zero
        if abs(augmented_matrix[i][i]) < 1e-12:
            raise ValueError("Matrix is singular or nearly singular.")
        
        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = augmented_matrix[j][i] / augmented_matrix[i][i]
            for k in range(i, n + 1):
                augmented_matrix[j][k] -= factor * augmented_matrix[i][k]
    
    # Back substitution
    coefficients = [0] * n
    for i in range(n - 1, -1, -1):
        coefficients[i] = (augmented_matrix[i][-1] - sum(augmented_matrix[i][j] * coefficients[j] for j in range(i + 1, n))) / augmented_matrix[i][i]
    
    return coefficients


def format_polynomial(coefficients):
    """Format the polynomial for display."""
    terms = []
    for i, coeff in enumerate(coefficients):
        if abs(coeff) > 1e-12:  # Ignore very small coefficients
            power = len(coefficients) - i - 1
            if power == 0:
                terms.append(f"{coeff:.15g}")
            elif power == 1:
                terms.append(f"{coeff:.15g}x")
            else:
                terms.append(f"{coeff:.15g}x^{power}")
    return terms
