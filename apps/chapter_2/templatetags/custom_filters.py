from django import template

register = template.Library()

@register.filter(name='get_range')
def get_range(value):
    return range(value)
