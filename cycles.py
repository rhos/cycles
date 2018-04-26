import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import svg.path as svg

DPI = math.pi * 2
COEFF_FILE = "drawing"
SVG_FILE = "{0}.svg".format(COEFF_FILE)

def integrate(func,start,end,dx=0.01):
    i=start
    area=0
    while i<=end:
        area+=func(i)*dx
        i+=dx
    return area

def read_path(filename):
    with open(filename) as f:
        shape=f.read()
    path=shape.split("<g")[1].split("<path")[1].split(' d="')[1].split('"')[0]
    path=svg.parse_path(path)
    return path

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

def calc_coeffs(path, start, end):
    coeffs=[]
    for i in range(start,end):
        coef = integrate(lambda a: path.point(a/DPI) * math.e**(-1j*i*a), 0, DPI) / DPI
        coeffs.append((i,coef))
    return coeffs

def save_coeffs(coeffs, filename) :
    with open(filename, 'w') as f:
        for coeff in coeffs:
            f.write("{0} {1} {2}\n".format(coeff[0], coeff[1].real, coeff[1].imag))

def read_coeffs(filename) :
    coeffs=[]
    with open(filename, 'r') as f:
        content = f.readlines()
    for line in content:
        split = line.split()
        coeffs.append((int(split[0]), complex(float(split[1]), float(split[2]))))
    return coeffs

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

if os.path.exists(COEFF_FILE):
    coeffs = read_coeffs(COEFF_FILE)
else:
    path = read_path(SVG_FILE)
    print("calculating offset...")
    offset = integrate(lambda t:path.point(t),0, 1) 
    path = translate_path(path, -offset)
    print("calculating coeffs...")
    coeffs = calc_coeffs(path,-50,50)
    save_coeffs(coeffs, COEFF_FILE)
coeffs.sort(key=lambda x:1/abs(x[1]))
xcoords,ycoords, cycles = get_coord(coeffs)

fig = plt.figure()
ax = plt.axes(xlim=(-80, 80), ylim=(-80, 80))
ax.set_aspect(1)
line, = ax.plot([], [], lw=2)
circles = []
line.set_data([], [])
for c in cycles[0]:
    circle = plt.Circle((c[0].real, c[0].imag), c[2], fc='y')
    circle.set_facecolor('none')
    circle.set_edgecolor('red')
    circles.append(circle)
    ax.add_patch(circle)
def init():
    return circles + [line]

def animate(i):
    x = xcoords[0:i+1]
    y = ycoords[0:i+1]
    line.set_data(x, y)
    if i < len(cycles):
        for j,c in enumerate(circles):
            c.radius = cycles[i][j][2]
            c.center = (cycles[i][j][0].real, cycles[i][j][0].imag)

    return circles + [line]

anim = anim.FuncAnimation(fig, animate, init_func=init,
                               frames=2000, interval=20, blit=True)

#plt.show()
anim.save('{0}.mp4'.format(SVG_FILE), fps=30, 
          extra_args=['-vcodec', 'h264', 
                      '-pix_fmt', 'yuv420p'])