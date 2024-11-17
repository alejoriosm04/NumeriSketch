import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render

def newton_int(request):
    initial_x = [-1, 0, 3, 4]
    initial_y = [15.5, 3, 8, 1]

    context = {
        'x_values': None,
        'y_values': None,
        'polynomial': None,
        'polynomial_latex': None,
        'divided_differences': [],
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

            context['x_values'] = x_values
            context['y_values'] = y_values

            polynomial, polynomial_latex, divided_differences = newton_interpolation(x_values, y_values)
            context['polynomial'] = polynomial
            context['polynomial_latex'] = polynomial_latex
            context['divided_differences'] = divided_differences
            context['points'] = list(zip(x_values, y_values))

            graph_paths = plot_newton(x_values, y_values, polynomial)
            context.update(graph_paths)

        except ValueError as e:
            context['error'] = str(e)
        except Exception as e:
            context['error'] = f"Error inesperado: {e}"

    return render(request, 'newton_int.html', context)

def divided_differences_table(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:,0] = y
    for j in range(1,n):
        for i in range(n-j):
            coef[i][j] = (coef[i+1][j-1]-coef[i][j-1])/(x[i+j]-x[i])
    return coef[0,:], coef  # Retorna los coeficientes y la tabla completa

def newton_polynomial(coef, x_data):
    n = len(coef)
    p = np.poly1d([coef[0]])
    for i in range(1, n):
        term = np.poly1d([1])
        for j in range(i):
            term = np.polymul(term, np.poly1d([1, -x_data[j]]))
        p = np.polyadd(p, coef[i]*term)
    return p

def generate_polynomial_latex(coef, x_data):
    terms = []
    for i in range(len(coef)):
        term = f"{coef[i]:.2f}"
        for j in range(i):
            xj = x_data[j]
            if xj == 0:
                term += "(x)"
            elif xj < 0:
                term += f"(x + {abs(xj):g})"
            else:
                term += f"(x - {xj:g})"
        terms.append(term)
    polynomial_latex = " + ".join(terms)
    return polynomial_latex

def newton_interpolation(x, y):
    coef, table = divided_differences_table(x, y)
    polynomial = newton_polynomial(coef, x)
    polynomial_latex = generate_polynomial_latex(coef, x)
    # Procesar la tabla para mostrarla en el template
    n = len(x)
    table_display = []
    for i in range(n):
        row = [f"{x[i]:.4f}"]
        for j in range(n - i):
            value = table[i][j]
            row.append(f"{value:.4f}")
        # Rellenar con cadenas vacías
        for _ in range(i):
            row.append('')
        table_display.append(row)
    return polynomial, polynomial_latex, table_display

def plot_newton(x, y, polynomial, png_path='static/graphs/newton_plot.png', svg_path='static/graphs/newton_plot.svg'):
    plt.figure(figsize=(8, 6))
    x_range = np.linspace(min(x) - 1, max(x) + 1, 500)
    y_range = np.polyval(polynomial, x_range)

    plt.plot(x_range, y_range, label="Polinomio de Newton", color='blue')
    plt.scatter(x, y, color='red', label="Puntos dados")
    plt.title("Interpolación de Newton")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    plt.savefig(png_path, format='png')
    plt.savefig(svg_path, format='svg')
    plt.close()

    return {'graph_png': png_path, 'graph_svg': svg_path}
