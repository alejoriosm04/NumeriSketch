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
        'error': None,
        'graph': None,
        'input': None
    }

    if request.method == 'POST':
        try:
            # Obtener y convertir los valores de entrada
            x_values = list(map(float, request.POST.get('x_values').split(',')))
            y_values = list(map(float, request.POST.get('y_values').split(',')))

            if len(x_values) != len(y_values):
                raise ValueError("Las listas de x e y deben tener la misma longitud.")
            if len(x_values) > 8:
                raise ValueError("No se permiten más de 8 datos.")

            # Crear una lista de tuplas (x, y)
            paired_values = list(zip(x_values, y_values))

            # Calcular el polinomio interpolante
            polynomial = vandermonde_method(x_values, y_values)

            # Usar la función para formatear el polinomio de forma legible
            #formatted_polynomial = format_polynomial(polynomial)
            #context['polynomial'] = formatted_polynomial  # Usar el polinomio formateado
            context['polynomial'] = polynomial  # Usar el polinomio formateado
            context['input'] = True

            # Generar y guardar la gráfica
            graph_path = plot_polynomial(polynomial, x_values, y_values)
            context['graph'] = graph_path
            context['paired_values'] = paired_values  # Pasar los valores emparejados al contexto

        except Exception as e:
            context['error'] = f"Error: {str(e)}"

    return render(request, 'vandermonde.html', context)

def plot_polynomial(polynomial, x_values, y_values):
    """
    Grafica el polinomio y los puntos de datos originales, y guarda la imagen como SVG.
    
    Args:
        polynomial (np.poly1d): Polinomio interpolante.
        x_values (list): Lista de valores x de los puntos.
        y_values (list): Lista de valores y correspondientes a cada x.

    Returns:
        str: Ruta del archivo SVG guardado.
    """
    # Crear valores de x para la gráfica
    x_plot = np.linspace(min(x_values), max(x_values), 500)
    y_plot = polynomial(x_plot)

    # Generar gráfica
    plt.figure()
    plt.plot(x_plot, y_plot, label="Interpolación de Vandermonde")
    plt.scatter(x_values, y_values, color="red", label="Puntos")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)

    # Guardar la imagen en SVG
    svg_path = os.path.join("static", "graphs", "vandermonde_interpolation.svg")
    plt.savefig(svg_path, format="svg")

    png_path = os.path.join("static", "graphs", "vandermonde_interpolation.png")
    plt.savefig(png_path, format="png")

    plt.close()
    
    return svg_path

    
def format_polynomial(polynomial):
    """
    Formatea un objeto np.poly1d en una cadena de texto legible como un polinomio.
    
    Args:
        polynomial (np.poly1d): Objeto polinómico creado por np.poly1d.
    
    Returns:
        str: Representación del polinomio en formato legible.
    """
    terms = []
    for i, coeff in enumerate(polynomial.coefficients):
        if coeff != 0:
            power = len(polynomial.coefficients) - i - 1
            if power == 0:
                terms.append(f"{coeff:.4f}")
            elif power == 1:
                terms.append(f"{coeff:.4f}x")
            else:
                terms.append(f"{coeff:.4f}x^{power}")
    
    return " + ".join(terms)


def vandermonde_method(x_values, y_values):
    """
    Genera el polinomio interpolante utilizando la matriz de Vandermonde.
    
    Args:
        x_values (list): Lista de valores x de los puntos.
        y_values (list): Lista de valores y correspondientes a cada x.

    Returns:
        np.poly1d: Objeto de tipo polinómico.
    """
    n = len(x_values)
    # Crear la matriz de Vandermonde
    V = np.vander(x_values, n)
    # Resolver el sistema para obtener los coeficientes
    coefficients = np.linalg.solve(V, y_values)
    # Crear el polinomio a partir de los coeficientes
    polynomial = np.poly1d(coefficients)
    
    # Retornar el objeto np.poly1d para usarlo en la gráfica
    return polynomial

