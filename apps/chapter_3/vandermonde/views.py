from django.shortcuts import render
from django.http import HttpResponseRedirect
import os
import numpy as np
import matplotlib.pyplot as plt

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

            polynomial = vandermonde_method(x_values, y_values)
            formatted_polynomial = format_polynomial(polynomial)
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

    
def format_polynomial(polynomial):

    terms = []
    for i, coeff in enumerate(polynomial.coefficients):
        if coeff != 0:
            print(coeff)
            power = len(polynomial.coefficients) - i - 1
            if power == 0:
                terms.append(f"{coeff}")    
            elif power == 1:
                terms.append(f"{coeff}x")
            else:
                terms.append(f"{coeff}x^{power}")
    
    return terms


def vandermonde_method(x_values, y_values):

    n = len(x_values)
    V = np.vander(x_values, n)

    coefficients = np.linalg.solve(V, y_values)
    polynomial = np.poly1d(coefficients)
    
    return polynomial

