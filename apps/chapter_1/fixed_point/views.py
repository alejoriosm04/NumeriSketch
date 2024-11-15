import os
from django.http import HttpResponseRedirect
from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usar el backend Agg para evitar problemas con la GUI


def generate_graph(x0, fx, gx, s, safe_dict):
    x_vals = np.linspace(x0 - 2, x0 + 2, 400)
    f_vals = [eval(fx, {"x": val, "math": math,
                   "__builtins__": {}}, safe_dict) for val in x_vals]
    # gx_vals = [eval(gx, {"x": valg, "math": math, "__builtins__": {}}, safe_dict) for valg in x_vals]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, f_vals, label='f(x)')
    # plt.plot(x_vals, gx_vals, label='g(x)')
    plt.scatter([s], [eval(fx, {"x": s, "math": math, "__builtins__": {
                }}, safe_dict)], color='red', zorder=5, label='Solution Point')
    plt.title('Graph of f(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

     # Ensure the directory exists
    graph_dir = os.path.join('static', 'graphs')
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)

    # Save the figure
    graph_path = os.path.join(graph_dir, 'fixed_point.png')
    plt.savefig(graph_path)

    # return os.path.join('graphs', 'graph.png')

    # Save the figure in SVG format
    svg_path = os.path.join(graph_dir, 'fixed_point.svg')
    plt.savefig(svg_path, format='svg')

    plt.close()

    return {
        'png': os.path.join('graphs', 'fixed_point.png'),
        'svg': os.path.join('graphs', 'fixed_point.svg')
    }


def fixed_point(request):
    context = {
        'x0': 0,
        'tol': 0,
        'niter': 0,
        'fx': '',
        'gx': '',
        'error_type': 'absolute',  # Valor por defecto
        'msg': [],
        'table': None,
        'error': True,
        'graph': None  # Path to the graph image
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
        'abs': abs,
        'factorial': math.factorial,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'log2': math.log2,
        'log10': math.log10,
        'gamma': math.gamma,
        'lgamma': math.lgamma
    }

    if request.method == 'POST':
        try:
            context['x0'] = float(request.POST.get('x0'))
            context['tol'] = float(request.POST.get('tol'))
            context['niter'] = int(request.POST.get('niter'))
            context['fx'] = request.POST.get('fx').replace('^', '**')
            context['gx'] = request.POST.get('gx').replace('^', '**')
            # Obtener el tipo de error seleccionado
            context['error_type'] = request.POST.get('error_type')

            x = context['x0']
            f = eval(context['fx'], {
                     "x": x, "math": math, "__builtins__": {}}, safe_dict)
            fn = []
            xn = []
            E = []
            N = []
            c = 0
            Error = 100
            fn.append(f)
            xn.append(x)
            E.append(Error)
            fe = None
            N.append(c)
            s = None

            while Error > context['tol'] and fe != 0 and c < context['niter']:
                x = eval(context['gx'], {
                         "x": x, "math": math, "__builtins__": {}}, safe_dict)
                fe = eval(context['fx'], {
                          "x": x, "math": math, "__builtins__": {}}, safe_dict)

                fn.append(fe)
                xn.append(x)
                c +=1

                if context['error_type'] == "relative":  
                    Error = abs((xn[c] - xn[c - 1]) / xn[c])
                else:  # Cálculo del error absoluto
                    Error = abs(xn[c] - xn[c - 1])

                N.append(c)
                E.append(Error)

            if fe == 0:
                s = x
                context['msg'].append(f"{s} es raíz de f(x)")
                context['error'] = False
            elif Error < context['tol']:
                s = x
                context['root'] = s
                context['msg'].append(f"{s} es una aproximación de una raíz de f(x) con una tolerancia de {context['tol']}")
                context['error'] = False
            else:
                context['msg'].append(f"Fracaso en {context['niter']} iteraciones.")
                
            if(s is not None):  
                context['graph'] = generate_graph(context['x0'], context['fx'], context['gx'], s, safe_dict)

        except Exception as e:
            context['msg'].append(f"Error: {str(e)}")
        
        data = {
            "iteration": N,
            "x_n": xn,
            "fx_n": fn,
            "error": E
        }
        df = pd.DataFrame(data)
        context['table'] = df.to_dict(orient='records')
        print("Table: ", context['table'])

    return render(request, 'fixed_point.html', context)
