import os
import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.http import HttpResponseRedirect

def vandermonde(request):
    initial_x = [0, 1, 2, 3]
    initial_y = [0, 1, 4, 9]

    context = {
        'x_values': None,
        'y_values': None,
        'polynomial': None,
        'polynomial_latex': None,
        'error': None,
        'graph_png': None,
        'graph_svg': None,
        'input': None,
        'paired_values': zip(initial_x, initial_y),
    }

    if request.method == 'POST':
        try:
            # Obtener listas de x e y desde el formulario
            x_values = [float(val) for val in request.POST.getlist('x[]')]
            y_values = [float(val) for val in request.POST.getlist('y[]')]

            # Validaciones
            if len(x_values) != len(y_values):
                raise ValueError("Las listas de x e y deben tener la misma longitud.")
            if len(x_values) < 2:
                raise ValueError("Debes ingresar al menos dos puntos para realizar la interpolación.")
            if len(x_values) > 8:
                raise ValueError("No se permiten más de 8 puntos para evitar problemas de rendimiento y precisión.")
            if len(x_values) != len(set(x_values)):
                raise ValueError("Los valores de x deben ser únicos para evitar divisiones por cero.")

            context['x_values'] = x_values
            context['y_values'] = y_values
            context['paired_values'] = list(zip(x_values, y_values))

            # Calcular el polinomio de interpolación de Vandermonde
            polynomial, coefficients = vandermonde_method(x_values, y_values)

            # Formatear el polinomio para visualización con LaTeX
            polynomial_latex = format_polynomial_latex(coefficients)
            context['polynomial'] = coefficients
            context['polynomial_latex'] = polynomial_latex

            # Generar la gráfica
            graph_paths = plot_polynomial(polynomial, x_values, y_values)
            context['graph_png'] = graph_paths['png']
            context['graph_svg'] = graph_paths['svg']

            context['input'] = True

        except ValueError as ve:
            context['error'] = str(ve)
        except Exception as e:
            context['error'] = f"Error inesperado: {e}"

    return render(request, 'vandermonde.html', context)


def plot_polynomial(polynomial, x_values, y_values, png_path='static/graphs/vandermonde_interpolation.png', svg_path='static/graphs/vandermonde_interpolation.svg'):
    """Genera y guarda la gráfica de la interpolación de Vandermonde."""
    plt.figure(figsize=(8, 6))
    x_plot = np.linspace(min(x_values) - 1, max(x_values) + 1, 500)
    y_plot = polynomial(x_plot)

    plt.plot(x_plot, y_plot, label="Interpolación de Vandermonde", color='blue')
    plt.scatter(x_values, y_values, color='red', label="Puntos dados")
    plt.title("Interpolación de Vandermonde")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    # Guardar la gráfica como PNG
    plt.savefig(png_path, format='png')
    # Guardar la gráfica como SVG
    plt.savefig(svg_path, format='svg')
    plt.close()

    return {'png': png_path, 'svg': svg_path}


def vandermonde_method(x_values, y_values):
    """Calcula el polinomio de interpolación de Vandermonde y devuelve una función polinómica."""
    n = len(x_values)

    # Construir la matriz de Vandermonde
    V = [[x**(n-i-1) for i in range(n)] for x in x_values]

    # Resolver el sistema V * coeficientes = y_values
    coefficients = gauss_solve(V, y_values)

    # Definir el polinomio como una función
    def polynomial(x):
        return sum(c * (x**i) for i, c in enumerate(reversed(coefficients)))

    return polynomial, coefficients


def gauss_solve(matrix, values):
    """Resuelve un sistema de ecuaciones lineales Ax = b usando eliminación gaussiana."""
    n = len(matrix)

    # Augmentar la matriz con el vector de valores
    augmented_matrix = [row + [val] for row, val in zip(matrix, values)]

    # Eliminación hacia adelante
    for i in range(n):
        # Encontrar la fila pivote y realizar el intercambio
        pivot_row = max(range(i, n), key=lambda r: abs(augmented_matrix[r][i]))
        augmented_matrix[i], augmented_matrix[pivot_row] = augmented_matrix[pivot_row], augmented_matrix[i]

        # Verificar si el pivote es cero
        if abs(augmented_matrix[i][i]) < 1e-12:
            raise ValueError("La matriz es singular o casi singular.")

        # Eliminar las entradas debajo del pivote
        for j in range(i + 1, n):
            factor = augmented_matrix[j][i] / augmented_matrix[i][i]
            for k in range(i, n + 1):
                augmented_matrix[j][k] -= factor * augmented_matrix[i][k]

    # Sustitución hacia atrás
    coefficients = [0] * n
    for i in range(n - 1, -1, -1):
        coefficients[i] = (augmented_matrix[i][-1] - sum(augmented_matrix[i][j] * coefficients[j] for j in range(i + 1, n))) / augmented_matrix[i][i]

    return coefficients


def format_polynomial_latex(coefficients):
    """Formatea el polinomio para su visualización en LaTeX."""
    terms = []
    degree = len(coefficients) - 1
    for i, coef in enumerate(coefficients):
        power = degree - i
        if abs(coef) < 1e-12:
            continue  # Ignorar coeficientes muy pequeños
        coef_str = f"{coef:.2f}"
        if power == 0:
            term = f"{coef_str}"
        elif power == 1:
            term = f"{coef_str}x"
        else:
            term = f"{coef_str}x^{{{power}}}"
        terms.append(term)
    polynomial_latex = " + ".join(terms)
    return polynomial_latex
