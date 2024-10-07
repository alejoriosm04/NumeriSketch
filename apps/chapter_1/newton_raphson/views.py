import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp  # SymPy para derivadas simbólicas
from io import BytesIO
import base64
from django.shortcuts import render

def newton_raphson(request):
    context = {}

    # Diccionario seguro para evaluar expresiones matemáticas
    safe_dict = {
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'pi': math.pi,
        'e': math.e,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'exp': math.exp,
        'sqrt': math.sqrt,
        'abs': abs,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'gamma': math.gamma,
        'lgamma': math.lgamma
    }

    if request.method == 'POST':
        try:
            # Obtener valores del formulario
            x0 = float(request.POST.get('x0', ''))
            tol = float(request.POST.get('tol', ''))
            niter = int(request.POST.get('niter', ''))
            fx = request.POST.get('fx', '').replace('^', '**')
            dfx = request.POST.get('dfx', '').replace('^', '**')
            error_type = request.POST.get('error_type', '')

            # Guardar el valor original de x0 para mostrarlo en el formulario después
            original_x0 = x0

            # Si no se ingresó la derivada, calcularla usando SymPy
            x = sp.Symbol('x')
            fx_sympy = sp.sympify(fx)
            if not dfx:
                dfx_sympy = sp.diff(fx_sympy, x)
                dfx = str(dfx_sympy)

            # Inicializar variables para iteraciones
            iteration = 0
            xn = [x0]
            fn = [eval(fx, {"x": x0, "math": math}, safe_dict)]
            errors = [100]  # La primera iteración no tiene error
            error = None
            root = None
            unrounded_root = None  # Guardar la raíz sin redondeo

            # Iterar usando el método de Newton-Raphson
            while iteration < niter:
                f_value = eval(fx, {"x": x0, "math": math}, safe_dict)
                df_value = eval(dfx, {"x": x0, "math": math}, safe_dict)

                if df_value == 0:
                    context['msg'] = [f"La derivada es cero en x = {x0}, el método no puede continuar."]
                    break

                # Fórmula del método de Newton-Raphson
                x1 = x0 - (f_value / df_value)

                # Calcular el error según el tipo de error seleccionado
                if iteration >= 0:
                    if error_type == "relativo":
                        error = abs((x1 - x0) / x1)
                    else:
                        error = abs(x1 - x0)

                # Guardar valores de la iteración
                xn.append(x1)
                fn.append(eval(fx, {"x": x1, "math": math}, safe_dict))
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
            else:
                context['msg'] = [f"No se encontró la raíz en {niter} iteraciones."]
                context['error'] = True

            # Generar gráfico
            fig, ax = plt.subplots()
            x_vals = np.linspace(xn[0] - 2, xn[0] + 2, 400)
            y_vals = [eval(fx, {"x": val, "math": math}, safe_dict) for val in x_vals]
            ax.plot(x_vals, y_vals, label='f(x)')
            ax.axhline(0, color='gray', lw=1)  # Línea en y = 0

            # Mostrar la raíz si se encontró
            if unrounded_root is not None:
                ax.axvline(unrounded_root, color='red', linestyle='--', label='Raíz encontrada')

            ax.legend()
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')

            context['graph'] = img_base64

            # Guardar valores del formulario en el contexto
            context.update({
                'x0': original_x0,  # Mantener el valor original de x0 ingresado por el usuario
                'tol': tol,
                'niter': niter,
                'fx': fx,
                'dfx': dfx,
                'error_type': error_type,
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
