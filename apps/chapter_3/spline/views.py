import numpy as np
import matplotlib.pyplot as plt
from django.shortcuts import render

def spline_view(request):
    initial_x = [-1, 0, 3, 4]
    initial_y = [15.5, 3, 8, 1]

    context = {
        'x_values': None,
        'y_values': None,
        'spline_segments': [],
        'graph_png': None,
        'graph_svg': None,
        'error': None,
        'points': zip(initial_x, initial_y),
    }

    if request.method == 'POST':
        try:
            x_values = [float(val) for val in request.POST.getlist('x[]')]
            y_values = [float(val) for val in request.POST.getlist('y[]')]
            spline_type = request.POST.get('spline_type', 'lineal')  # Obtener el tipo de spline

            if len(x_values) < 2 or len(y_values) < 2:
                raise ValueError("Debes ingresar al menos dos puntos para realizar la interpolación.")

            if len(x_values) != len(set(x_values)):
                raise ValueError("Los valores de X o Y no deben repetirse para evitar divisiones por cero.")

            context['x_values'] = x_values
            context['y_values'] = y_values
            context['points'] = list(zip(x_values, y_values))

            spline_segments = spline_interpolation(x_values, y_values, spline_type)
            context['spline_segments'] = spline_segments

            graph_paths = plot_spline(x_values, y_values, spline_segments)
            context.update(graph_paths)

        except ValueError as e:
            context['error'] = str(e)
        except Exception as e:
            context['error'] = f"Error inesperado: {e}"

    return render(request, 'spline.html', context)

def spline_interpolation(x, y, spline_type):
    if spline_type == 'lineal':
        return linear_spline(x, y)
    elif spline_type == 'cuadratico':
        return quadratic_spline(x, y)
    else:
        raise ValueError("Tipo de spline no válido.")

def linear_spline(x, y):
    n = len(x)
    segments = []
    for i in range(n - 1):
        m = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) 
        segment = f"S_{{{i}}}(x) = {y[i]:.2f} + {m:.2f}(x - {x[i]:.2f})"
        segments.append(segment)
    return segments

def quadratic_spline(x, y):
    n = len(x)
    segments = []
    a = y
    b = []
    c = [0] * (n - 1) 

    for i in range(n - 1):
        b.append((y[i + 1] - y[i]) / (x[i + 1] - x[i]))

    for i in range(1, n - 1):
        c[i] = 2 * ((b[i] - b[i - 1]) / (x[i + 1] - x[i - 1]))

    for i in range(n - 1):
        segment = f"S_{{{i}}}(x) = {a[i]:.2f} + {b[i]:.2f}(x - {x[i]:.2f}) + {c[i]:.2f}(x - {x[i]:.2f})^2"
        segments.append(segment)

    return segments


def evaluate_segment(segment, x_range):
    import re
    match = re.search(r'([\d\.\-]+)\s+\+\s+([\d\.\-]+)\(x\s+\-\s+([\d\.\-]+)\)(?:\s+\+\s+([\d\.\-]+)\(x\s+\-\s+[\d\.\-]+\)\^2)?', segment)
    if not match:
        raise ValueError(f"Formato de segmento inválido: {segment}")
    
    a = float(match.group(1))
    b = float(match.group(2))
    x0 = float(match.group(3))
    c = float(match.group(4)) if match.group(4) else 0

    return a + b * (x_range - x0) + c * (x_range - x0) ** 2


def plot_spline(x, y, spline_segments, png_path='static/graphs/spline_plot.png', svg_path='static/graphs/spline_plot.svg'):
    plt.figure(figsize=(8, 6))

    for i, segment in enumerate(spline_segments):
        x_range = np.linspace(x[i], x[i + 1], 100)
        y_range = evaluate_segment(segment, x_range)
        plt.plot(x_range, y_range, label=f'Segmento {i}')

    plt.scatter(x, y, color='red', label='Puntos originales')
    plt.title('Interpolación Spline')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.savefig(png_path)
    plt.savefig(svg_path)
    plt.close()

    return {'graph_png': png_path, 'graph_svg': svg_path}

