import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render
import time  # Importar el módulo time para generar timestamps

def spline_view(request):
    initial_x = [0, 1, 2, 3, 4]
    initial_y = [0, 1, 4, 9, 16]

    context = {
        'x_values': None,
        'y_values': None,
        'spline_type': 'lineal',  # Valor predeterminado
        'splines': [],
        'graph_png': None,
        'graph_svg': None,
        'error': None,
        'points': zip(initial_x, initial_y),
        'timestamp': None,  # Añadido para el cache busting
    }

    if request.method == 'POST':
        try:
            x_values = [float(val) for val in request.POST.getlist('x[]')]
            y_values = [float(val) for val in request.POST.getlist('y[]')]
            spline_type = request.POST.get('spline_type', 'lineal')

            if len(x_values) < 2 or len(y_values) < 2:
                raise ValueError("Debes ingresar al menos dos puntos para realizar la interpolación.")

            if len(x_values) != len(set(x_values)):
                raise ValueError("Los valores de X no deben repetirse para evitar problemas en el cálculo del spline.")

            if spline_type not in ['lineal', 'cuadratico']:
                raise ValueError("Tipo de spline no válido.")

            context['x_values'] = x_values
            context['y_values'] = y_values
            context['spline_type'] = spline_type
            context['points'] = list(zip(x_values, y_values))

            # Ordenar los puntos por x para evitar problemas en el cálculo del spline
            sorted_points = sorted(zip(x_values, y_values), key=lambda pair: pair[0])
            x_values, y_values = zip(*sorted_points)

            if spline_type == 'lineal':
                splines = linear_spline(x_values, y_values)
            else:
                splines = quadratic_spline(x_values, y_values)

            context['splines'] = splines

            # Generar un timestamp único para el cache busting
            timestamp = int(time.time())
            context['timestamp'] = timestamp

            graph_paths = plot_spline(x_values, y_values, splines, spline_type)
            context.update(graph_paths)

        except ValueError as e:
            context['error'] = str(e)
        except Exception as e:
            context['error'] = f"Error inesperado: {e}"

    return render(request, 'spline.html', context)


def linear_spline(x, y):
    n = len(x)
    splines = []

    for i in range(n - 1):
        xi, xi1 = x[i], x[i+1]
        yi, yi1 = y[i], y[i+1]

        # Coeficientes del spline lineal en el intervalo [xi, xi+1]
        m = (yi1 - yi) / (xi1 - xi)
        b = yi - m * xi

        # Representación del spline
        spline_equation = f"S_{i}(x) = {m:.2f}x + {b:.2f}, para x ∈ [{xi}, {xi1}]"
        splines.append(spline_equation)

    return splines


def quadratic_spline(x, y):
    n = len(x)
    splines = []

    # Número de splines cuadráticos es n - 1
    # Cada spline tiene la forma S_i(x) = a_i x^2 + b_i x + c_i

    # Construimos un sistema de ecuaciones lineales para resolver los coeficientes
    A = []
    B = []

    # Ecuaciones de paso: S_i(x_i) = y_i y S_i(x_i+1) = y_i+1
    for i in range(n - 1):
        xi, xi1 = x[i], x[i+1]
        row1 = [xi**2, xi, 1] + [0]*(3*(n-1)-(i+1)*3)
        row2 = [xi1**2, xi1, 1] + [0]*(3*(n-1)-(i+1)*3)
        if i > 0:
            row1 = [0]*3*i + row1
            row2 = [0]*3*i + row2
        A.append(row1)
        B.append(y[i])
        A.append(row2)
        B.append(y[i+1])

    # Ecuaciones de suavidad: S_i'(x_i+1) = S_i+1'(x_i+1)
    for i in range(n - 2):
        xi1 = x[i+1]
        row = [0]*3*i
        row += [2*xi1, 1, 0, -2*xi1, -1, 0]
        row += [0]*3*(n-2-i-1)
        A.append(row)
        B.append(0)

    # Ecuación adicional: Derivada en el primer punto
    # S_0'(x0) = (y1 - y0) / (x1 - x0)
    slope_initial = (y[1] - y[0]) / (x[1] - x[0])
    row_additional = [2*x[0], 1, 0] + [0]*(3*(n-1)-3)
    A.append(row_additional)
    B.append(slope_initial)

    # Ahora, el sistema tiene 3n - 3 ecuaciones y 3n -3 incógnitas
    A = np.array(A)
    B = np.array(B)
    coeffs = np.linalg.solve(A, B)

    # Extraer los coeficientes y construir las ecuaciones
    for i in range(n - 1):
        a = coeffs[3*i]
        b = coeffs[3*i + 1]
        c = coeffs[3*i + 2]
        xi, xi1 = x[i], x[i+1]
        spline_equation = f"S_{i}(x) = {a:.2f}x^2 + {b:.2f}x + {c:.2f}, para x ∈ [{xi}, {xi1}]"
        splines.append(spline_equation)

    return splines


def plot_spline(x, y, splines, spline_type, png_path='static/graphs/spline_plot.png', svg_path='static/graphs/spline_plot.svg'):
    plt.figure(figsize=(8, 6))

    # Graficar los splines
    x_vals = np.linspace(min(x), max(x), 500)

    if spline_type == 'lineal':
        for i in range(len(x) - 1):
            xi, xi1 = x[i], x[i+1]
            m = (y[i+1] - y[i]) / (x[i+1] - x[i])
            b = y[i] - m * x[i]
            x_range = x_vals[(x_vals >= xi) & (x_vals <= xi1)]
            y_range = m * x_range + b
            # Añadir etiqueta solo una vez para la leyenda
            label = "Spline Lineal" if i == 0 else ""
            plt.plot(x_range, y_range, color='blue', label=label)
    else:
        # Obtener los coeficientes nuevamente para graficar
        n = len(x)
        A = []
        B = []

        for i in range(n - 1):
            xi, xi1 = x[i], x[i+1]
            row1 = [xi**2, xi, 1] + [0]*(3*(n-1)-(i+1)*3)
            row2 = [xi1**2, xi1, 1] + [0]*(3*(n-1)-(i+1)*3)
            if i > 0:
                row1 = [0]*3*i + row1
                row2 = [0]*3*i + row2
            A.append(row1)
            B.append(y[i])
            A.append(row2)
            B.append(y[i+1])

        for i in range(n - 2):
            xi1 = x[i+1]
            row = [0]*3*i
            row += [2*xi1, 1, 0, -2*xi1, -1, 0]
            row += [0]*3*(n-2-i-1)
            A.append(row)
            B.append(0)

        # Ecuación adicional: Derivada en el primer punto
        slope_initial = (y[1] - y[0]) / (x[1] - x[0])
        row_additional = [2*x[0], 1, 0] + [0]*(3*(n-1)-3)
        A.append(row_additional)
        B.append(slope_initial)

        A = np.array(A)
        B = np.array(B)
        coeffs = np.linalg.solve(A, B)

        for i in range(n - 1):
            a = coeffs[3*i]
            b = coeffs[3*i + 1]
            c = coeffs[3*i + 2]
            xi, xi1 = x[i], x[i+1]
            x_range = x_vals[(x_vals >= xi) & (x_vals <= xi1)]
            y_range = a * x_range**2 + b * x_range + c
            # Añadir etiqueta solo una vez para la leyenda
            label = "Spline Cuadrático" if i == 0 else ""
            plt.plot(x_range, y_range, color='green', label=label)

    plt.scatter(x, y, color='red', label="Puntos dados")
    plt.title(f"Interpolación Spline {spline_type.capitalize()}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    plt.savefig(png_path, format='png')
    plt.savefig(svg_path, format='svg')
    plt.close()

    return {'graph_png': png_path, 'graph_svg': svg_path}
