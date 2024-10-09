import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
from django.shortcuts import render
import math

def falseposition_view(request):
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
            # Obtener los valores del formulario
            xi = float(request.POST.get('xi', ''))
            xs = float(request.POST.get('xs', ''))
            tol = float(request.POST.get('tol', ''))
            niter = int(request.POST.get('niter', ''))
            fun = request.POST.get('fun', '')

            # Reemplazar '^' con '**' para la sintaxis de potencia en Python
            fun = fun.replace('^', '**')

            precision_type = request.POST.get('precision_type', '')
            precision_value = int(request.POST.get('precision_value', ''))

            # Ejecutar el método de bisección
            fm, E, root, iterations = falseposition_method(xi, xs, tol, niter, fun, safe_dict)

            # Definir el tipo de error a mostrar en la tabla
            error_type = 'Error Relativo' if precision_type == 'significant_figures' else 'Error Absoluto'

            # Generar gráfico
            fig, ax = plt.subplots()
            x_vals = np.linspace(xi, xs, 100)
            y_vals = [eval(fun, {"x": val, "math": math}, safe_dict) for val in x_vals]
            ax.plot(x_vals, y_vals)
            ax.axhline(0, color='gray', lw=1)  # línea en y = 0

            # Si se encuentra la raíz, ajustar con la precisión solicitada
            if root is not None:
                if precision_type == 'significant_figures':
                    fm = [round_to_significant_figures(f, precision_value) for f in fm]
                    root = round_to_significant_figures(root, precision_value)
                    E = [round_to_significant_figures(e, precision_value) for e in E]
                elif precision_type == 'decimal_places':
                    fm = [round(f, precision_value) for f in fm]
                    root = round(root, precision_value)
                    E = [round(e, precision_value) for e in E]

                ax.axvline(root, color='red', linestyle='--')

            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')

            # Crear contexto con los resultados
            context = {
                'fx': fun,
                'msg': ['Cálculo completado'],
                'graph': img_base64,
                'table': [{'iteration': i, 'x_n': fm[i], 'fx_n': fm[i], 'error': E[i]} for i in range(iterations)],
                'error_type': error_type,
                'root': root,
                # Valores del formulario para que se mantengan en la vista
                'xi': xi,
                'xs': xs,
                'tol': tol,
                'niter': niter,
                'fun': fun,
                'precision_type': precision_type,
                'precision_value': precision_value,
            }

        except Exception as e:
            context['msg'] = [f"Error: {str(e)}"]

    return render(request, 'falseposition.html', context)

def round_to_significant_figures(value, sig_figs):
    if value == 0:
        return 0
    else:
        return round(value, sig_figs - int(np.floor(np.log10(abs(value)))) - 1)

def falseposition_method(Xi, Xs, Tol, Niter, Fun, safe_dict):
    fm = []
    E = []
    x = Xi
    fi = eval(Fun, {"x": Xi, "math": math}, safe_dict)
    x = Xs
    fs = eval(Fun, {"x": Xs, "math": math}, safe_dict)

    if fi == 0:
        return [], [], Xi, 0
    elif fs == 0:
        return [], [], Xs, 0
    elif fs * fi < 0:
        c = 0
        Xm = Xs - (fs * (Xi - Xs)) / (fi - fs)
        x = Xm
        fe = eval(Fun, {"x": Xm, "math": math}, safe_dict)
        fm.append(fe)
        E.append(100)

        while E[c] > Tol and fe != 0 and c < Niter:
            if fi * fe < 0:
                Xs = Xm
                x = Xs
                fs = eval(Fun, {"x": Xs, "math": math}, safe_dict)
            else:
                Xi = Xm
                x = Xi
                fs = eval(Fun, {"x": Xi, "math": math}, safe_dict)

            Xa = Xm
            Xm = Xs - (fs * (Xi - Xs)) / (fi - fs)
            x = Xm
            fe = eval(Fun, {"x": Xm, "math": math}, safe_dict)
            fm.append(fe)
            Error = abs(Xm - Xa)
            E.append(Error)
            c += 1

        if fe == 0 or Error < Tol:
            return fm, E, Xm, c
        else:
            return fm, E, Xm, Niter
    else:
        return [], [], None, None
