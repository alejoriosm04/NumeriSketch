from django import template

register = template.Library()

@register.filter(name='get_range')
def get_range(value):
    return range(value)

@register.filter(name='index')
def index(List, i):
    try:
        return List[int(i)]
    except (IndexError, TypeError, ValueError):
        return ''
