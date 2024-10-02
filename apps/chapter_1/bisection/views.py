import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
from django.shortcuts import render

def bisection_view(request):
    context = {}

    if request.method == 'POST':

        xi = float(request.POST.get('xi', ''))
        xs = float(request.POST.get('xs', ''))
        tol = float(request.POST.get('tol', ''))
        niter = int(request.POST.get('niter', ''))
        fun = request.POST.get('fun', '')

        fun = fun.replace('^', '**')

        precision_type = request.POST.get('precision_type', '')
        precision_value = int(request.POST.get('precision_value', ''))

        fm, E, root, iterations = bisection_method(xi, xs, tol, niter, fun)

        error_type = 'Error Relativo' if precision_type == 'significant_figures' else 'Error Absoluto'

        fig, ax = plt.subplots()
        x_vals = np.linspace(xi, xs, 100)
        y_vals = [eval(fun) for x in x_vals]
        ax.plot(x_vals, y_vals)
        ax.axhline(0, color='gray', lw=1)  # line in y = 0

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

        context = { # for image
            'fx': fun,
            'msg': ['Calculation complete'],
            'graph': img_base64,
            'table': [{'iteration': i, 'x_n': fm[i], 'fx_n': fm[i], 'error': E[i]} for i in range(iterations)],
            'error_type': error_type,
            'root': root,
            # form values
            'xi': xi,
            'xs': xs,
            'tol': tol,
            'niter': niter,
            'fun': fun,
            'precision_type': precision_type,
            'precision_value': precision_value,
        }

    return render(request, 'bisection.html', context)

# helper function to round to significant figures
def round_to_significant_figures(value, sig_figs):
    if value == 0:
        return 0
    else:
        return round(value, sig_figs - int(np.floor(np.log10(abs(value)))) - 1)


def bisection_method(Xi, Xs, Tol, Niter, Fun):
    fm = []
    E = []
    x = Xi
    fi = eval(Fun)
    x = Xs
    fs = eval(Fun)

    if fi == 0:
        return [], [], Xi, 0
    elif fs == 0:
        return [], [], Xs, 0
    elif fs * fi < 0:
        c = 0
        Xm = (Xi + Xs) / 2
        x = Xm
        fe = eval(Fun)
        fm.append(fe)
        E.append(100)

        while E[c] > Tol and fe != 0 and c < Niter:
            if fi * fe < 0:
                Xs = Xm
                x = Xs
                fs = eval(Fun)
            else:
                Xi = Xm
                x = Xi
                fs = eval(Fun)

            Xa = Xm
            Xm = (Xi + Xs) / 2
            x = Xm
            fe = eval(Fun)
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
