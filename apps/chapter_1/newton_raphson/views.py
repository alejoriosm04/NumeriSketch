# apps/chapter_1/newton_raphson/views.py
import sympy as sp
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usar el backend Agg para evitar problemas con la GUI
import matplotlib.pyplot as plt
import numpy as np
from django.shortcuts import render
import os

def generate_graph(x0, fx, dfx, s, safe_dict):
    x_vals = np.linspace(x0 - 2, x0 + 2, 400)

    # Graficar f(x) y f'(x)
    f_vals = [eval(fx, {"x": val, "math": math, "__builtins__": {}}, safe_dict) for val in x_vals]
    dfx_vals = [eval(dfx, {"x": val, "math": math, "__builtins__": {}}, safe_dict) for val in x_vals]

    plt.figure(figsize=(10, 6))
    
    # Graficar f(x)
    plt.plot(x_vals, f_vals, label='f(x)', color='blue')
    
    # Graficar f'(x)
    plt.plot(x_vals, dfx_vals, label="f'(x)", color='green')
    
    # Punto solución
    plt.scatter([s], [eval(fx, {"x": s, "math": math, "__builtins__": {}}, safe_dict)], color='red', zorder=5, label='Punto Solución')

    # Título de la gráfica incluyendo la función f(x)
    plt.title(f"Gráfica de f(x) = {fx} y f'(x) = {dfx}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    # Guardar el gráfico como PNG y SVG
    graph_dir = os.path.join('static', 'graphs')
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    graph_path = os.path.join(graph_dir, 'graph_newton.png')
    plt.savefig(graph_path)

    svg_path = os.path.join(graph_dir, 'graph_newton.svg')
    plt.savefig(svg_path, format='svg')

    plt.close()

    return {
        'png': os.path.join('graphs', 'graph_newton.png'),
        'svg': os.path.join('graphs', 'graph_newton.svg')
    }

def newton_raphson(request):
    context = {
        'x0': 0,
        'tol': 0,
        'niter': 0,
        'fx': '',
        'dfx': '',  # Derivada de f(x)
        'error_type': 'absoluto',  # Tipo de error por defecto
        'msg': [],
        'table': None,
        'error': True,
        'graph': None,
        'summary': ''
    }

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
            # Obtener datos del formulario
            context['x0'] = float(request.POST.get('x0'))
            context['tol'] = float(request.POST.get('tol'))
            context['niter'] = int(request.POST.get('niter'))
            context['fx'] = request.POST.get('fx').replace('^', '**')
            context['dfx'] = request.POST.get('dfx', '')  # Se permite que dfx sea vacío
            context['error_type'] = request.POST.get('error_type')  # Obtener tipo de error

            # Si no se proporcionó la derivada, la calculamos automáticamente
            if not context['dfx']:
                # Usamos SymPy para derivar la función
                x = sp.Symbol('x')
                fx_sympy = sp.sympify(context['fx'].replace('**', '^'))  # Convertimos a SymPy compatible
                dfx_sympy = sp.diff(fx_sympy, x)  # Derivamos
                context['dfx'] = str(dfx_sympy).replace('^', '**')  # Convertimos de nuevo la derivada a string

                context['msg'].append(f"La derivada automática es: f'(x) = {context['dfx']}")

            # Inicializar valores
            x0 = context['x0']
            tol = context['tol']
            niter = context['niter']
            fx = context['fx']
            dfx = context['dfx']
            error_type = context['error_type']

            error = None
            iteration = 0
            xn = [x0]
            fn = [eval(fx, {"x": x0, "math": math}, safe_dict)]
            errors = [100]

            # Resumen de la operación
            context['summary'] = f"Usando el método de Newton-Raphson para resolver f(x) = {fx} con derivada f'(x) = {dfx}, valor inicial x0 = {x0}, tolerancia = {tol}, máximo de iteraciones = {niter}, y error de tipo {error_type}."

            # Método de Newton-Raphson
            while iteration < niter:
                f_value = eval(fx, {"x": x0, "math": math}, safe_dict)
                df_value = eval(dfx, {"x": x0, "math": math}, safe_dict)

                if df_value == 0:
                    context['msg'].append(f"La derivada es cero en x = {x0}, el método no puede continuar.")
                    break

                # Fórmula de Newton-Raphson
                x1 = x0 - (f_value / df_value)

                # Calcular el error según el tipo seleccionado
                if iteration >= 0:
                    if error_type == 'absoluto':
                        error = abs(x1 - x0)
                    elif error_type == 'relativo':
                        error = abs(x1 - x0) / abs(x1) if x1 != 0 else float('inf')

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

            # Crear la tabla de iteraciones
            data = {
                "iteration": list(range(0, iteration + 1)),
                "x_n": xn,
                "f_xn": fn,
                "error": errors
            }

            df = pd.DataFrame(data)
            context['table'] = df.to_dict(orient='records')

            # Evaluar el resultado final
            if error is not None and error < tol:
                context['msg'].append(f"Raíz aproximada encontrada: {x0} con un error de {error}.")
                context['error'] = False
            else:
                context['msg'].append(f"No convergió en {niter} iteraciones.")

            # Generar gráfico de f(x) y f'(x)
            context['graph'] = generate_graph(context['x0'], context['fx'], context['dfx'], x0, safe_dict)

        except Exception as e:
            context['msg'].append(f"Error: {str(e)}")

    return render(request, 'newton_raphson.html', context)
