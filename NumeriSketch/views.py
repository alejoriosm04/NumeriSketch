from django.shortcuts import render

# Vista para la página de inicio
def home_view(request):
    return render(request, 'home.html')
