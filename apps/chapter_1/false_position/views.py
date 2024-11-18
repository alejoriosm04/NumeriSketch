import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import numpy as np
from django.shortcuts import render
import math
import re

def falseposition_graph(fun, xi, xs, root=None, png_path='static/graphs/falseposition.png', svg_path='static/graphs/falseposition.svg'):
    x_vals = np.linspace(xi, xs, 400)
    y_vals = [eval(fun, {"x": val, **safe_math()}) for val in x_vals]

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
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh
    }

def falseposition_view(request):
    context = {}
    if request.method == 'POST':
        try:
            xi = float(request.POST.get('xi', '').strip())
            xs = float(request.POST.get('xs', '').strip())
            tol = float(request.POST.get('tol', '').strip())
            niter = int(request.POST.get('niter', '').strip())
            fun = request.POST.get('fun', '').strip()

            fun = preprocess_function(fun)

            try:
                eval(fun, {"x": 1, **safe_math()})
            except Exception as eval_error:
                raise ValueError(f"Expresión no válida: {eval_error}")

            root = None
            try:
                fm, E, root, iterations = falseposition_method(xi, xs, tol, niter, fun, safe_math())
                context['msg'] = ['Cálculo completado']
                context['table'] = [{'iteration': i, 'x_n': fm[i], 'fx_n': fm[i], 'error': E[i]} for i in range(len(E))]
                context['root'] = root
            except ValueError as ve:
                context['msg'] = [f"Advertencia: {str(ve)}"]

            graph_paths = falseposition_graph(fun, xi, xs, root)

            context.update({
                'fx': fun,
                'graph_png': graph_paths['png'],
                'graph_svg': graph_paths['svg'],
                'error_type': 'Error Absoluto',
                'xi': xi,
                'xs': xs,
                'tol': tol,
                'niter': niter,
                'fun': fun,
            })

        except ValueError as e:
            context.update({
                'msg': [f"Error de entrada: {str(e)}"],
                'xi': request.POST.get('xi', ''),
                'xs': request.POST.get('xs', ''),
                'tol': request.POST.get('tol', ''),
                'niter': request.POST.get('niter', ''),
                'fun': request.POST.get('fun', ''),
            })
        except Exception as e:
            context.update({
                'msg': [f"Error inesperado: {str(e)}"],
                'xi': request.POST.get('xi', ''),
                'xs': request.POST.get('xs', ''),
                'tol': request.POST.get('tol', ''),
                'niter': request.POST.get('niter', ''),
                'fun': request.POST.get('fun', ''),
            })

    return render(request, 'falseposition.html', context)

def falseposition_method(Xi, Xs, Tol, Niter, Fun, safe_dict):
    fm = []
    E = []
    x = Xi
    fi = eval(Fun, {"x": Xi}, safe_dict)
    x = Xs
    fs = eval(Fun, {"x": Xs}, safe_dict)

    if fi == 0:
        return [Xi], [0], Xi, 1
    elif fs == 0:
        return [Xs], [0], Xs, 1
    elif fs * fi < 0:
        c = 0
        Xm = Xs - (fs * (Xi - Xs)) / (fi - fs)
        x = Xm
        fe = eval(Fun, {"x": Xm}, safe_dict)
        fm.append(fe)
        E.append(abs(Xs - Xi))

        while E[c] > Tol and fe != 0 and c < Niter:
            if fi * fe < 0:
                Xs = Xm
            else:
                Xi = Xm
            Xa = Xm
            Xm = Xs - (fs * (Xi - Xs)) / (fi - fs)
            x = Xm
            fe = eval(Fun, {"x": Xm}, safe_dict)
            fm.append(fe)
            Error = abs(Xm - Xa)
            E.append(Error)
            c += 1

        if fe == 0 or E[c] < Tol:
            return fm, E, Xm, c + 1
        else:
            return fm, E, Xm, c + 1
    else:
        raise ValueError("El intervalo proporcionado no contiene una raíz. Verifique que la función cambie de signo en el intervalo seleccionado.")
