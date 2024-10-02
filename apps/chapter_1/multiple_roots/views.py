import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from django.shortcuts import render

def generate_multiple_roots_graph(x0, fx, dfx, ddfx, s, safe_dict):
    # Define the x values for plotting
    x_vals = np.linspace(x0 - 2, x0 + 2, 400)

    # Evaluate f(x), f'(x), and f''(x)
    f_vals = [eval(fx, {"x": val, "math": math}, safe_dict) for val in x_vals]
    dfx_vals = [eval(dfx, {"x": val, "math": math}, safe_dict) for val in x_vals]
    ddfx_vals = [eval(ddfx, {"x": val, "math": math}, safe_dict) for val in x_vals]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot f(x)
    plt.plot(x_vals, f_vals, label='f(x)', color='blue')

    # Plot f'(x)
    plt.plot(x_vals, dfx_vals, label="f'(x)", color='green')

    # Plot f''(x)
    plt.plot(x_vals, ddfx_vals, label="f''(x)", color='orange')

    # Highlight the root found by the method (if provided)
    if s is not None:
        plt.scatter([s], [eval(fx, {"x": s, "math": math}, safe_dict)], color='red', zorder=5, label='Punto Raíz')

    # Title and labels
    plt.title(f"Gráfica de f(x) = {fx}, f'(x) = {dfx}, y f''(x) = {ddfx}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    # Define the directory to save the plot
    graph_dir = os.path.join('static', 'graphs')
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    # Save the plot as a PNG and SVG
    graph_path_png = os.path.join(graph_dir, 'graph_multiple_roots.png')
    graph_path_svg = os.path.join(graph_dir, 'graph_multiple_roots.svg')

    plt.savefig(graph_path_png)
    plt.savefig(graph_path_svg, format='svg')

    plt.close()

    # Return the paths to the saved images
    return {
        'png': os.path.join('graphs', 'graph_multiple_roots.png'),
        'svg': os.path.join('graphs', 'graph_multiple_roots.svg')
    }

import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp  # SymPy para derivadas simbólicas
from django.shortcuts import render

def multiple_roots(request):
    context = {
        'x0': 0,
        'tol': 0,
        'niter': 0,
        'fx': '',
        'dfx': '',  # First derivative
        'ddfx': '',  # Second derivative
        'error_type': '',  # Error type (absolute or relative)
        'msg': [],
        'table': None,
        'error': True,
        'graph': None,
        'summary': ''
    }

    # Diccionario seguro para evitar funciones no permitidas en eval
    safe_dict = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'pi': math.pi,
        'e': math.e,
        'log': math.log,
        'exp': math.exp,
        'sqrt': math.sqrt,
        'abs': abs
    }

    if request.method == 'POST':
        try:
            # Obtener los valores del formulario
            context['x0'] = float(request.POST.get('x0'))
            context['tol'] = float(request.POST.get('tol'))
            context['niter'] = int(request.POST.get('niter'))
            context['fx'] = request.POST.get('fx').replace('^', '**')
            context['dfx'] = request.POST.get('dfx').replace('^', '**')  # First derivative
            context['ddfx'] = request.POST.get('ddfx').replace('^', '**')  # Second derivative
            context['error_type'] = request.POST.get('error_type')  # Error type (absolute or relative)

            # Verificar si se introdujeron las derivadas
            if not context['dfx'] or not context['ddfx']:
                x = sp.Symbol('x')  # Definir la variable simbólica
                fx_sympy = sp.sympify(context['fx'])  # Convertir f(x) a una expresión simbólica
                
                # Calcular la primera derivada si no se proporcionó
                if not context['dfx']:
                    dfx_sympy = sp.diff(fx_sympy, x)  # Derivada de f(x)
                    context['dfx'] = str(dfx_sympy)  # Convertir a string para su uso posterior

                # Calcular la segunda derivada si no se proporcionó
                if not context['ddfx']:
                    ddfx_sympy = sp.diff(fx_sympy, x, 2)  # Segunda derivada de f(x)
                    context['ddfx'] = str(ddfx_sympy)  # Convertir a string para su uso posterior

            # Inicializar valores
            x0 = context['x0']
            tol = context['tol']
            niter = context['niter']
            fx = context['fx']
            dfx = context['dfx']
            ddfx = context['ddfx']
            error = None
            iteration = 0
            xn = [x0]
            fn = [eval(fx, {"x": x0, "math": math}, safe_dict)]
            errors = [100]  # Primera iteración no tiene error

            # Resumen de la operación
            context['summary'] = f"Usando el Método de Raíces Múltiples para f(x) = {fx}, f'(x) = {dfx}, f''(x) = {ddfx}, valor inicial x0 = {x0}, tolerancia = {tol}, y max iteraciones = {niter}, tipo de error = {context['error_type']}."

            # Iteraciones
            while iteration < niter:
                f_value = eval(fx, {"x": x0, "math": math}, safe_dict)
                df_value = eval(dfx, {"x": x0, "math": math}, safe_dict)
                ddf_value = eval(ddfx, {"x": x0, "math": math}, safe_dict)

                if df_value == 0 and ddf_value == 0:
                    context['msg'].append(f"La primera y segunda derivadas son cero en x = {x0}, el método no puede continuar.")
                    break

                # Fórmula del Método de Raíces Múltiples
                x1 = x0 - (f_value * df_value) / ((df_value ** 2) - (f_value * ddf_value))

                # Calcular el error según el tipo seleccionado
                if iteration >= 0:
                    if context['error_type'] == "relativo":
                        error = abs((x1 - x0) / x1)  # Error relativo
                    else:
                        error = abs(x1 - x0)  # Error absoluto

                # Guardar los valores de la iteración
                xn.append(x1)
                fn.append(eval(fx, {"x": x1, "math": math}, safe_dict))
                errors.append(error)

                # Actualizar para la siguiente iteración
                x0 = x1
                iteration += 1

                # Verificar convergencia
                if error is not None and error < tol:
                    break

            # Crear tabla de iteraciones
            data = {
                "iteration": list(range(0, iteration + 1)),
                "x_n": xn,
                "f_xn": fn,
                "error": errors
            }

            df = pd.DataFrame(data)
            context['table'] = df.to_dict(orient='records')

            # Mensaje final basado en la convergencia
            if error is not None and error < tol:
                context['msg'].append(f"Raíz aproximada encontrada: {x0} con un error de {error}.")
                context['error'] = False
            else:
                context['msg'].append(f"No convergió en {niter} iteraciones.")

            # Puedes agregar la generación de gráficos aquí, similar al método de Newton-Raphson
            # Ejemplo:
            context['graph'] = generate_multiple_roots_graph(context['x0'], context['fx'], context['dfx'], context['ddfx'], x0, safe_dict)

        except Exception as e:
            context['msg'].append(f"Error: {str(e)}")

    return render(request, 'multiple_roots.html', context)
