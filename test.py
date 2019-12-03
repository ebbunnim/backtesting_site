from django.http import HttpResponse
import io
import matplotlib.pyplot as plt
import numpy as np

def setPlt():
    numPts = 50
    x = [random.random() for n in range(numPts)]
    y = [random.random() for n in range(numPts)]
    sz = 2 ** (10*np.random.rand(numPts))
    plt.scatter(x, y, s=sz, alpha=0.5)

def pltToSvg():
    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    s = buf.getvalue()
    buf.close()
    return s

def get_svg(request):
    setPlt() # create the plot
    svg = pltToSvg() # convert plot to SVG
    plt.cla() # clean up plt so it can be re-used
    response = HttpResponse(svg, content_type='image/svg+xml')
    return response


# from django.http import HttpResponse
# import random
# import numpy as np
# from matplotlib.backends.backend_svg import FigureCanvas
# from matplotlib.figure import Figure

# def getFig():
#     fig=Figure(facecolor='w')
#     ax=fig.add_subplot(111)
#     numPts = 50
#     x = [random.random() for n in range(numPts)]
#     y = [random.random() for n in range(numPts)]
#     sz = 2 ** (10*np.random.rand(numPts))
#     ax.scatter(x, y, s=sz, alpha=0.5)
#     return fig

# def get_svg(request):
#     fig = getFig()
#     response = HttpResponse(content_type='image/svg+xml')
#     FigureCanvas(fig).print_svg(response)
#     return response