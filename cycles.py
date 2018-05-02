import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import svg.path as svg
import xml.etree.ElementTree as etree
import itertools

DPI = math.pi * 2
NAME = "Emblem_of_Moscow_Aviation_Institute"
COEFF_FILE = "{0}.coeff".format(NAME)
SVG_FILE = "{0}.svg".format(NAME)

colors = ['blue','green','red','cyan','magenta','yellow','black']
colorsNum = len(colors)

def integrate(func,start,end,dx=0.01):
    i=start
    area=0
    while i<=end:
        area+=func(i)*dx
        i+=dx
    return area

def read_paths(filename):
    tree = etree.parse(filename)
    for pathnode in tree.findall(".//{http://www.w3.org/2000/svg}path"):
        yield mirror_path(svg.parse_path(pathnode.attrib['d']))

def translate_path(path,offset):
    displaced=svg.Path()
    for curve in path:
        if type(curve) == svg.Line:
            newCurve = svg.Line(curve.start+offset, curve.end+offset)
        elif type(curve) == svg.CubicBezier:
            newCurve = svg.CubicBezier(curve.start+offset, curve.control1+offset, curve.control2+offset, curve.end+offset)
        elif type(curve) == svg.QuadraticBezier:
            newCurve = svg.QuadraticBezier(curve.start+offset, curve.control+offset, curve.end+offset)
        elif type(curve) == svg.Arc:
            newCurve = svg.Arc(curve.start+offset, curve.radius, curve.rotation, curve.arc, curve.sweep, curve.end+offset)
        displaced.append(newCurve)
    displaced.closed = path.closed
    return displaced

def invertImg(num):
    return complex(num.real, -1*num.imag)

def mirror_path(path):
    displaced=svg.Path()
    for curve in path:
        if type(curve) == svg.Line:
            newCurve = svg.Line(invertImg(curve.start), invertImg(curve.end))
        elif type(curve) == svg.CubicBezier:
            newCurve = svg.CubicBezier(invertImg(curve.start), invertImg(curve.control1), invertImg(curve.control2), invertImg(curve.end))
        elif type(curve) == svg.QuadraticBezier:
            newCurve = svg.QuadraticBezier(invertImg(curve.start), invertImg(curve.control), invertImg(curve.end))
        elif type(curve) == svg.Arc:
            newCurve = svg.Arc(invertImg(curve.start), curve.radius, curve.rotation, curve.arc, curve.sweep, invertImg(curve.end))
        displaced.append(newCurve)
    displaced.closed = path.closed
    return displaced

def calc_coeffs(path, start, end):
    coeffs=[]
    for i in range(start,end):
        coef = integrate(lambda a: path.point(a/DPI) * math.e**(-1j*i*a), 0, DPI) / DPI
        coeffs.append((i,coef))
    return coeffs

def save_coeffs(coeffsList, filename) :
    with open(filename, 'w') as f:
        for coeffs in coeffsList:
            for coeff in coeffs:
                f.write("{0} {1} {2};".format(coeff[0], coeff[1].real, coeff[1].imag))
            f.write('\n')

def read_coeffs(filename) :
    coeffsList=[]
    with open(filename, 'r') as f:
        content = f.readlines()
    for line in content:
        coeffs = []
        for coeffLine in line.split(';'):
            split = coeffLine.split()
            if len(split) == 3:
                coeffs.append((int(split[0]), complex(float(split[1]), float(split[2]))))
        if len(coeffs) > 0:
            coeffsList.append(coeffs)
    return coeffsList

def get_coord(coeffs):
    coordX = []
    coordY = []
    cycles = []
    for t in np.arange(0, DPI, 0.01):
        sum = complex(0,0)
        circles = []
        for coeff in coeffs:
            pos = coeff[1] * math.e**(1j*coeff[0]*t)
            circles.append((sum, pos, abs(coeff[1])))
            sum += pos
        coordX.append(sum.real)
        coordY.append(sum.imag)
        cycles.append(circles)
    return coordX, coordY, cycles

coeffsList = []
if os.path.exists(COEFF_FILE):
    coeffsList = read_coeffs(COEFF_FILE)
else:
    paths = list(read_paths(SVG_FILE))
    print("calculating offset...")
    offset = integrate(lambda t:paths[0].point(t),0, 1) 
    paths = [translate_path(path, -offset) for path in paths]
    print("calculating coeffs...")
    for path in paths:
        coeffsList.append(calc_coeffs(path,-50,50))
    save_coeffs(coeffsList, COEFF_FILE)

fig = plt.figure()
ax = plt.axes(xlim=(-500, 500), ylim=(-500, 500))
ax.set_aspect(1)
ax.plot([], [], lw=2)


circles = []
lines = []
data = []
for i,coeffs in enumerate(coeffsList):
    #coeffs.sort(key=lambda x:1/abs(x[1]))
    xcoords, ycoords, cycles = get_coord(coeffs)
    data.append((xcoords, ycoords, cycles))

    lc = []
    for c in cycles[0]:
        circle = plt.Circle((c[0].real, c[0].imag), c[2], fc='y')
        circle.set_facecolor('none')
        circle.set_edgecolor(colors[i % colorsNum])
        lc.append(circle)
        ax.add_patch(circle)
    circles.append(lc)

    line = plt.Line2D([], [])
    lines.append(line)
    ax.add_patch(line)

def init():
    return list(itertools.chain(*circles)) + lines

def animate(i):
    for ind,d in enumerate(data):
        xdata = d[0]
        ydata = d[1]
        cycles = d[2]

        x = xdata[0:i+1]
        y = ydata[0:i+1]
        lines[ind].set_data(x, y)

        if i < len(cycles):
            for j,c in enumerate(circles[ind]):
                c.radius = cycles[i][j][2]
                c.center = (cycles[i][j][0].real, cycles[i][j][0].imag)

    return list(itertools.chain(*circles)) + lines

anim = anim.FuncAnimation(fig, animate, init_func=init,
                               frames=2000, interval=20, blit=True)

#plt.show()
anim.save('{0}.mp4'.format(SVG_FILE), fps=30, 
          extra_args=['-vcodec', 'h264', 
                      '-pix_fmt', 'yuv420p'])