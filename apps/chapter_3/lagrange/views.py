import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render

def lagrange_view(request):
    initial_x = [-1, 0, 3, 4]
    initial_y = [15.5, 3, 8, 1]

    context = {
        'x_values': None,
        'y_values': None,
        'polynomial': None,
        'polynomial_latex': None,
        'lagrange_terms': [],
        'graph_png': None,
        'graph_svg': None,
        'error': None,
        'points': zip(initial_x, initial_y),
    }

    if request.method == 'POST':
        try:
            x_values = [float(val) for val in request.POST.getlist('x[]')]
            y_values = [float(val) for val in request.POST.getlist('y[]')]

            if len(x_values) < 2 or len(y_values) < 2:
                raise ValueError("Debes ingresar al menos dos puntos para realizar la interpolación.")

            if len(x_values) != len(set(x_values)):
                raise ValueError("Los valores de X no deben repetirse para evitar divisiones por cero.")

            if len(x_values) > 8:
                raise ValueError("Solo puedes ingresar hasta 8 puntos.")

            context['x_values'] = x_values
            context['y_values'] = y_values

            polynomial, polynomial_latex, lagrange_terms = lagrange_interpolation(x_values, y_values)
            context['polynomial'] = polynomial
            context['polynomial_latex'] = polynomial_latex
            context['lagrange_terms'] = lagrange_terms
            context['points'] = list(zip(x_values, y_values))

            graph_paths = plot_lagrange(x_values, y_values, polynomial)
            context.update(graph_paths)

        except ValueError as e:
            context['error'] = str(e)
        except Exception as e:
            context['error'] = f"Error inesperado: {e}"

    return render(request, 'lagrange.html', context)




def lagrange_interpolation(x, y):
    n = len(x)
    polynomial = np.poly1d([0])
    lagrange_terms = []  

    for i in range(n):
        Li = np.poly1d([1])  
        denominator = 1  

        for j in range(n):
            if i != j:
                Li = np.polymul(Li, np.poly1d([1, -x[j]]))  
                denominator *= (x[i] - x[j]) 

        term = (Li / denominator) * y[i]  
        polynomial += term  

        numerator_latex = " \\cdot ".join([f"(x {'+' if -x[j] < 0 else '-'} {abs(x[j]):g})" for j in range(n) if j != i])
        lagrange_terms.append(f"L_{{ {i} }}(x) = \\frac{{{numerator_latex}}}{{{denominator:.2f}}}")


    polynomial_latex = " + ".join([
        f"{coef:.2f}x^{len(polynomial.c) - i - 1}" if len(polynomial.c) - i - 1 > 1 else (
            f"{coef:.2f}x" if len(polynomial.c) - i - 1 == 1 else f"{coef:.2f}"
        )
        for i, coef in enumerate(polynomial.c)
    ])

    return polynomial, polynomial_latex, lagrange_terms



def plot_lagrange(x, y, polynomial, png_path='static/graphs/lagrange_plot.png', svg_path='static/graphs/lagrange_plot.svg'):
    plt.figure(figsize=(8, 6))
    x_range = np.linspace(min(x) - 1, max(x) + 1, 500)
    y_range = np.polyval(polynomial, x_range)

    plt.plot(x_range, y_range, label="Polinomio de Lagrange", color='blue')
    plt.scatter(x, y, color='red', label="Puntos dados")
    plt.title("Interpolación de Lagrange")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    plt.savefig(png_path, format='png')
    plt.savefig(svg_path, format='svg')
    plt.close()

    return {'graph_png': png_path, 'graph_svg': svg_path}

