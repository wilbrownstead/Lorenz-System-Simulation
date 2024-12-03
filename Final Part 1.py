import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import scipy.special
from mpl_toolkits import mplot3d
from matplotlib import animation
from scipy.signal import argrelextrema
import array as arr




def lorenz(t,y,sigma, b, r):
    
    #t is the time to evaluate the system
    #We are using y as a catch all 3d state vector of the form [x,y,z]
    
    final = np.array([
        sigma * (y[1]-y[0]),
        r * y[0] - y[1] - y[2] * y[0],
        y[0] * y[1] - b * y[2]
        ])
    return final


def dor_p(lorenz,t0,y0,sigma,b,r, tol, dt):
    k1 = lorenz(t0,y0,sigma,b,r)*dt
    k2 = lorenz(t0 + (1/5)*dt,y0 + (1/5)*k1,sigma,b,r)*dt
    k3 = lorenz(t0 + (3/10)*dt,y0 + (3/40)*k1 + (9/40)*k2,sigma,b,r)*dt
    k4 = lorenz(t0 + (4/5)*dt,y0 + (44/45)*k1 - (56/15)*k2 + (32/9)*k3,sigma,b,r)*dt
    k5 = lorenz(t0 + (8/9)*dt,y0 +  (19372/6561)*k1 - (25360/2187)*k2 + (64448/6561)*k3 - (212/729)*k4,sigma,b,r)*dt
    k6 = lorenz(t0 + dt,y0 + (9017/3168)*k1 - (355/33)*k2 + (46732/5247)*k3 + (49/176)*k4 - (5013/18656)*k5,sigma,b,r)*dt
    k7 = lorenz(t0 + dt,y0 + (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6,sigma,b,r)*dt
    
    yk = y0 + (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6
    # diff = min(abs((71/57600)*k1 - (71/166695)*k3 + (71/1920)*k4 - (17253/339200)*k5 + (22/525)*k6 - (1/40)*k7))
    # if diff < tol:
    #     diff = tol
    # s = ((dt*tol)/(2*diff))**(1/5)
    # if s > 1:
    #     dtk = dt*s
    #     # if count < 100:
    #     #     print('greater')
    #     #     print(dt)
    #     #     print(dtk)
    #     if dtk > dt*5:
    #         dtk = dt
    #     # if count < 100:
    #     #     print(dtk)
    #     dt = dtk
    # else:
    #     dtk = dt*s
    #     #if count < 100:
    #         # print('less')
    #         # print(dt)
    #         # print(dtk)
    #     if dtk < dt*0.1:
    #         dtk = 0.001
    #     #if count < 100:
    #         # print(dtk)
    #     dt = dtk
    
    return yk

    
def fehlberg(lorenz,t0,y0,sigma,b,r,tol,dt):
    k1 = lorenz(t0,y0,sigma,b,r)*dt
    k2 = lorenz(t0 + (1/4)*dt,y0 + (1/4)*k1,sigma,b,r)*dt
    k3 = lorenz(t0 + (3/8)*dt,y0 + (3/32)*k1 + (9/32)*k2,sigma,b,r)*dt
    k4 = lorenz(t0 + (12/13)*dt,y0 + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3,sigma,b,r)*dt
    k5 = lorenz(t0 + dt,y0 + (439/216)*k1 - (8)*k2 + (3680/513)*k3 - (845/4104)*k4,sigma,b,r)*dt
    k6 = lorenz(t0 + (1/2)*dt,y0 - (8/27)*k1 + (2)*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5,sigma,b,r)*dt

    
    yk = y0 + (25/216)*k1 + (1408/2565)*k3 + (2197/4101)*k4 - (1/5)*k5
    tk = y0 + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6
    diff = min(abs(tk-yk))
    if diff < tol:
        diff = tol
    #print(diff)
    s = 0.84*((tol*dt)/diff)**(0.25)
    #if count < 100:
        #print(s)
    if s > 1:
        dtk = dt*s
        # if count < 100:
        #     print('greater')
        #     print(dt)
        #     print(dtk)
        if dtk > dt*5:
            dtk = dt*5
        # if count < 100:
        #     print(dtk)
        dt = dtk
    else:
        dtk = dt*s
        #if count < 100:
            # print('less')
            # print(dt)
            # print(dtk)
        if dtk < dt*0.1:
            dtk = 0.001
        #if count < 100:
            # print(dtk)
        dt = dtk
    
        
    
    dt = s*dt
    
    return yk, dt

def rk4(lorenz,t0,y0,sigma,b,r,dt):
    k1 = lorenz(t0,y0,sigma,b,r)*dt
    k2 = lorenz(t0+(1/2)*dt,y0+(1/2)*k1,sigma,b,r)*dt
    k3 = lorenz(t0+(1/2)*dt, y0 + (1/2)*k2,sigma,b,r)*dt
    k4 = lorenz(t0+dt, y0+k3,sigma,b,r)*dt
    
    y = y0 + (1/6)*(k1+2*k2+2*k3+k4)
    return y

def rk45(lorenz,t0,y0,sigma,b,r, dt):
    k1 = lorenz(t0,y0,sigma,b,r)*dt
    k2 = lorenz(t0 + (1/5)*dt,y0 + (0.2)*k1,sigma,b,r)*dt
    k3 = lorenz(t0 + (3/10)*dt,y0 + (.075)*k1 + (.225)*k2,sigma,b,r)*dt
    k4 = lorenz(t0 + (3/5)*dt,y0 + (0.3)*k1 - (0.9)*k2 + (1.2)*k3,sigma,b,r)*dt
    k5 = lorenz(t0 + dt,y0 - (11/54)*k1 + (2.5)*k2 - (70/27)*k3 + (35/27)*k4,sigma,b,r)*dt
    k6 = lorenz(t0 + (3/4)*dt,y0 + (1631/55296)*k1 + (175/512)*k2 + (575/13824)*k3 + (44275/110592)*k4 - (253/4096)*k5,sigma,b,r)*dt
    
    yk = y0 + (37/378)*k1 + (250/621)*k3 + (125/594)*k4 - (512/1771)*k6
    # tk = y0 + (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6
    # diff = abs(tk-yk)
    
    # s = ((tol*dt)/(2*diff))**(1/4)
    # dt_new = min(s*dt)
    
    return yk



def main():
    
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("init_x",type=float,
                        help="Initial X Value")
    parser.add_argument("init_y",type=float,
                        help="Initial Y Value")
    parser.add_argument("init_z",type=float,
                        help="Initial Z Value")
    #parser.add_argument("sigma",type=float,
                       # help="Sigma Value")
    #parser.add_argument("b",type=float,
                       # help="B Value")
    parser.add_argument("r",type=float,
                        help="R Value")
    parser.add_argument("tolerance",type=float,
                        help="Final Tolerance")
    
    
    args       = parser.parse_args()
    init_x = args.init_x
    init_y = args.init_y
    init_z = args.init_z
    #Standard parameter values for Lorenz equations
    sig = 10
    b_lor = 8/3
    
    r_lor = args.r
    tol = args.tolerance
    
    #Create some saved list of values, we can convert to an array later
    saved_vals_1 = []
    saved_t_1 = []
    saved_vals_2 = []
    saved_t_2 = []
    saved_vals_3 = []
    saved_t_3 = []
    
    #Create initial conditions
    y0 = np.array([init_x, init_y, init_z])
    dt = 0.05
    
    #create a starting yk value
    # yk = y0
    
    # #What we want to do here is use our adaptive stepsize integral to get to roughly some defined time -> use a while loop
    # t = 0
    # count = 0
    
    # while t <= 10:
    # #add our values to our future arrays
    #     saved_vals_1.append(yk)
    #     saved_t_1.append(t)
                     
    #     t += dt
    #     yk,dt = dor_p(lorenz, t, yk, sig, b_lor, r_lor, tol, dt)
    #     #print(dt)
    #     count += 1
    
    
    #
    #The following while loops produce two manually shifted graphs for comparison's sake, they are commented out right now to allow for variable step size ODE integration 


    yk = y0
    t = 0
    dt = 0.005
    
    while t <= 50:
    #add our values to our future arrays
        saved_vals_1.append(yk)
        saved_t_1.append(t)
                     
        t += dt
        yk = dor_p(lorenz, t, yk, sig, b_lor, 28, tol, dt)
        #print(dt)
    
    yk = y0
    t = 0
    dt = 0.005
        
    while t <= 50:
    #add our values to our future arrays
        saved_vals_2.append(yk)
        saved_t_2.append(t)
                     
        t += dt
        yk = dor_p(lorenz, t, yk, sig, b_lor, 29, tol, dt)
        #print(dt)
    yk = y0
    t = 0
    dt = 0.005
    
    while t <= 50:
    #add our values to our future arrays
        saved_vals_3.append(yk)
        saved_t_3.append(t)
                     
        t += dt
        yk = dor_p(lorenz, t, yk, sig, b_lor, 30, tol, dt)
        #print(dt)
    
        
    saved_vals_1 = np.array(saved_vals_1)
    saved_t_1 = np.array(saved_t_1)
    saved_vals_2 = np.array(saved_vals_2)
    saved_t_2 = np.array(saved_t_2)
    saved_vals_3 = np.array(saved_vals_3)
    saved_t_3 = np.array(saved_t_3)
    
    #print(saved_vals_3)
    #USe these to create 2d graphs that they asked for

    x = saved_vals_1[:,0]
    y = saved_vals_1[:,1]
    z = saved_vals_1[:,2]
    x2 = saved_vals_2[:,0]
    y2 = saved_vals_2[:,1]
    z2 = saved_vals_2[:,2]
    x3 = saved_vals_3[:,0]
    y3 = saved_vals_3[:,1]
    z3 = saved_vals_3[:,2]
    print('Data done')
    
    
    fig = plt.figure('Standard View', figsize = (10,10), frameon = False)
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')

    # plotting
    #you can comment out different ax.plot things if you only need certain parts
    ax.plot(x, y, z, 'b', label = '212')
    #ax.plot(x2,y2,z2, 'g', label = '22.1')
    #ax.plot(x3,y3,z3, 'r', label = '22.2')
    ax.set_title('Lorenz System')
    ax.axes.set_xlim3d(left = min(x), right = max(x))
    ax.axes.set_ylim3d(bottom = min(y), top = max(y))
    ax.axes.set_zlim3d(bottom = min(z), top = max(z))
    ax.set_facecolor('white')
    ax._axis3don = False
    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.zaxis.set_ticklabels([])
    ax.legend()
    
    
    fig_an = plt.figure("Animated View", figsize = (10,10), frameon = False) 
    ax_an = fig_an.add_subplot(projection = '3d')
 
    line, = ax_an.plot(x[0:1], y[0:1], z[0:1], lw = 2, color = 'b', label = '28') 
    line_2, = ax_an.plot(x2[0:1], y2[0:1], z2[0:1], lw = 2, color = 'g', label = '29')
    line_3, = ax_an.plot(x3[0:1], y3[0:1], z3[0:1], lw = 2, color = 'r', label = '30') 
    
    #print(saved_vals[:,0])
        
    def animate(num, line, iterations):
        line.set_data([x[:num], y[:num]]) 
        line.set_3d_properties(z[:num])
        
        line_2.set_data([x2[:num], y2[:num]]) 
        line_2.set_3d_properties(z2[:num]) 
        
        line_3.set_data([x3[:num], y3[:num]]) 
        line_3.set_3d_properties(z3[:num]) 
        
        rotations = num*360*5/iterations
        ax_an.view_init(azim = rotations)
        return line,
    
    
    
    
    ax_an.axes.set_xlim3d(left = min(x), right = max(x))
    ax_an.axes.set_ylim3d(bottom = min(y), top = max(y))
    ax_an.axes.set_zlim3d(bottom = min(z), top = max(z))
    ax_an.set_facecolor('white')
    ax_an._axis3don = False
    ax_an.xaxis.set_ticklabels([])
    ax_an.yaxis.set_ticklabels([])
    ax_an.zaxis.set_ticklabels([])
    #ax_an.view_init(0, 45)
    ax_an.legend()
    
    iterations = len(saved_vals_1) - 1

    ani = animation.FuncAnimation(fig_an, animate, iterations,fargs = (line, iterations), interval=0.5, blit=False)
    ani.save('final.gif', fps = 100)
    plt.draw()
    
    
    
    #Lorenz Map of our fun solution stuff
    map_x = []
    map_y = []
    ext_z= []
    
    ext = argrelextrema(z, np.greater)
    for val in ext:
        for value in val:
            ext_z.append(value)
    print(ext_z)
    
    for i in range(0,len(ext_z)-2):
        print(ext_z[i])
        map_x.append(ext_z[i])
        map_y.append(ext_z[i+1])
        
    print(map_x)
    print(map_y)
    fig_map = plt.figure('Lorenz Map')
    plt.title('Lorenz Map')
    plt.plot(map_x, map_y, '.')
    plt.xlabel('z_n')
    plt.ylabel('z_n+1')
    plt.show()
    
    return



main()


