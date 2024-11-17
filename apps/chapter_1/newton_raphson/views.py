import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp  # SymPy para derivadas simbólicas
from django.shortcuts import render
import re

def preprocess_function(fun):
    """Preprocesa la función para convertir `^` a `**` y agregar multiplicaciones implícitas."""
    fun = fun.replace('^', '**')
    fun = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', fun)
    fun = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', fun)
    return fun

def safe_math():
    """Provee un entorno seguro con funciones matemáticas."""
    return {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'pi': np.pi,
        'e': np.e,
        'log': np.log,
        'log10': np.log10,
        'log2': np.log2,
        'exp': np.exp,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'asin': np.arcsin,
        'acos': np.arccos,
        'atan': np.arctan,
        'atan2': np.arctan2,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'gamma': sp.gamma,
        'lgamma': sp.loggamma
    }

def newton_graph(fun, xi, xf, root=None, png_path='static/graphs/newton_graph.png', svg_path='static/graphs/newton_graph.svg'):
    """Genera y guarda la gráfica de la función para el método de Newton-Raphson."""
    x_vals = np.linspace(xi, xf, 400)
    y_vals = []
    safe_dict = safe_math()
    for val in x_vals:
        try:
            y = eval(fun, {"x": val, **safe_dict})
        except:
            y = np.nan  # Maneja puntos donde la función no está definida
        y_vals.append(y)

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='f(x)', color='blue')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    if root is not None:
        plt.axvline(root, color='red', linestyle='--', label='Raíz aproximada')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title('Gráfica de f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()

    plt.savefig(png_path, format='png')
    plt.savefig(svg_path, format='svg')
    plt.close()

    return {'png': png_path, 'svg': svg_path}

def newton_raphson(request):
    context = {}

    if request.method == 'POST':
        try:
            # Obtener valores del formulario
            x0 = float(request.POST.get('x0', '').strip())
            tol = float(request.POST.get('tol', '').strip())
            niter = int(request.POST.get('niter', '').strip())
            fx = request.POST.get('fx', '').strip()
            dfx = request.POST.get('dfx', '').strip()
            error_type = request.POST.get('error_type', '').strip()

            # Preprocesar la función
            fx = preprocess_function(fx)
            if dfx:
                dfx = preprocess_function(dfx)

            # Verificar la función
            try:
                eval(fx, {"x": x0, **safe_math()})
            except Exception as eval_error:
                raise ValueError(f"Expresión no válida en f(x): {eval_error}")

            # Si no se ingresó la derivada, calcularla usando SymPy
            x_sym = sp.Symbol('x')
            fx_sympy = sp.sympify(fx)
            if not dfx:
                dfx_sympy = sp.diff(fx_sympy, x_sym)
                dfx = str(dfx_sympy)

            # Verificar la derivada
            try:
                eval(dfx, {"x": x0, **safe_math()})
            except Exception as eval_error:
                raise ValueError(f"Expresión no válida en f'(x): {eval_error}")

            # Inicializar variables para iteraciones
            iteration = 0
            xn = [x0]
            fn = [eval(fx, {"x": x0, **safe_math()})]
            errors = [None]  # La primera iteración no tiene error
            error = None
            root = None
            unrounded_root = None  # Guardar la raíz sin redondeo
            derivative_zero = False  # Bandera para derivada cero

            # Iterar usando el método de Newton-Raphson
            while iteration < niter:
                f_value = eval(fx, {"x": x0, **safe_math()})
                df_value = eval(dfx, {"x": x0, **safe_math()})

                if df_value == 0:
                    derivative_zero = True
                    context['msg'] = [f"La derivada es cero en x = {x0}. La derivada en el punto debe ser diferente de cero."]
                    break

                # Fórmula del método de Newton-Raphson
                x1 = x0 - (f_value / df_value)

                # Calcular el error según el tipo de error seleccionado
                if iteration >= 0:
                    if error_type == "relativo":
                        if x1 != 0:
                            error = abs((x1 - x0) / x1)
                        else:
                            error = abs(x1 - x0)
                    else:
                        error = abs(x1 - x0)

                # Guardar valores de la iteración
                xn.append(x1)
                fn.append(eval(fx, {"x": x1, **safe_math()}))
                errors.append(error)

                # Actualizar valor para la siguiente iteración
                x0 = x1
                iteration += 1

                # Verificar si se cumple la tolerancia
                if error is not None and error < tol:
                    unrounded_root = x1  # Guardar la raíz sin redondeo
                    root = x1
                    break

            # Crear tabla de iteraciones
            data = {
                "iteration": list(range(0, iteration + 1)),
                "x_n": [format_float_to_full(x) for x in xn],  # Mostrar todos los dígitos de x_n
                "f_xn": [format_float_to_full(f) for f in fn],  # Mostrar todos los dígitos de f_xn
                "error": [format_float_to_full(e) if e is not None else None for e in errors]  # Mostrar todos los dígitos del error
            }
            df = pd.DataFrame(data)
            context['table'] = df.to_dict(orient='records')

            # Mensajes de resultado
            if unrounded_root is not None:
                context['msg'] = [f"Raíz aproximada encontrada: {unrounded_root} con una tolerancia de {tol}."]
                context['error'] = False
            elif derivative_zero:
                context['error'] = True
                # No se agrega otro mensaje aquí porque ya se agregó anteriormente
            else:
                context['msg'] = [f"No se encontró la raíz en {niter} iteraciones."]
                context['error'] = True

            # Generar gráfico
            if len(xn) > 1:
                xi = min(xn)
                xf = max(xn)
            else:
                xi = x0 - 2
                xf = x0 + 2
            delta = (xf - xi) * 0.1
            xi -= delta
            xf += delta

            graph_paths = newton_graph(fx, xi, xf, root)

            context.update({
                'fx': fx,
                'dfx': dfx,
                'graph_png': graph_paths['png'],
                'graph_svg': graph_paths['svg'],
                'error_type': 'Error Relativo' if error_type == "relativo" else 'Error Absoluto',
                'x0': request.POST.get('x0', ''),
                'tol': tol,
                'niter': niter,
                'error_type_selected': error_type,
                'root': unrounded_root,  # Mostrar la raíz sin redondear
            })

        except Exception as e:
            context['msg'] = [f"Error: {str(e)}"]
            context['error'] = True

    return render(request, 'newton_raphson.html', context)

# Función auxiliar para mostrar el valor completo de un número flotante
def format_float_to_full(value):
    if value is None:
        return None
    return "{:.15f}".format(value).rstrip('0').rstrip('.') if '.' in "{:.15f}".format(value) else value
