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
import random
import wave
from playsound import playsound
from scipy.io.wavfile import write


def sender(t,y,sigma, b, r):
    
    #t is the time to evaluate the system
    #We are using y as a catch all 3d state vector of the form [x,y,z]
    
    final = np.array([
        sigma * (y[1]-y[0]),
        r * y[0] - y[1] - y[2] * y[0],
        y[0] * y[1] - b * y[2]
        ])
    return final

def reciever(t,y,sigma, b, r, X, i):
    
    #t is the time to evaluate the system
    #We are using y as a catch all 3d state vector of the form [x,y,z]
    
    final = np.array([
        sigma * (y[1]-y[0]),
        r * X[i] - y[1] - X[i] * y[2],
        X[i] * y[1] - b * y[2]
        ])
    return final



def dor_p(func,t0,y0,sigma,b,r, tol, dt):
    k1 = func(t0,y0,sigma,b,r)*dt
    k2 = func(t0 + (1/5)*dt,y0 + (1/5)*k1,sigma,b,r)*dt
    k3 = func(t0 + (3/10)*dt,y0 + (3/40)*k1 + (9/40)*k2,sigma,b,r)*dt
    k4 = func(t0 + (4/5)*dt,y0 + (44/45)*k1 - (56/15)*k2 + (32/9)*k3,sigma,b,r)*dt
    k5 = func(t0 + (8/9)*dt,y0 +  (19372/6561)*k1 - (25360/2187)*k2 + (64448/6561)*k3 - (212/729)*k4,sigma,b,r)*dt
    k6 = func(t0 + dt,y0 + (9017/3168)*k1 - (355/33)*k2 + (46732/5247)*k3 + (49/176)*k4 - (5013/18656)*k5,sigma,b,r)*dt
    k7 = func(t0 + dt,y0 + (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6,sigma,b,r)*dt
    
    yk = y0 + (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6
    diff = min(abs((71/57600)*k1 - (71/166695)*k3 + (71/1920)*k4 - (17253/339200)*k5 + (22/525)*k6 - (1/40)*k7))
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



def dor_p_rec(func,t0,y0,sigma,b,r, tol, dt, X, count):
    k1 = func(t0,y0,sigma,b,r,X,count)*dt
    k2 = func(t0 + (1/5)*dt,y0 + (1/5)*k1,sigma,b,r,X,count)*dt
    k3 = func(t0 + (3/10)*dt,y0 + (3/40)*k1 + (9/40)*k2,sigma,b,r,X,count)*dt
    k4 = func(t0 + (4/5)*dt,y0 + (44/45)*k1 - (56/15)*k2 + (32/9)*k3,sigma,b,r,X,count)*dt
    k5 = func(t0 + (8/9)*dt,y0 +  (19372/6561)*k1 - (25360/2187)*k2 + (64448/6561)*k3 - (212/729)*k4,sigma,b,r,X,count)*dt
    k6 = func(t0 + dt,y0 + (9017/3168)*k1 - (355/33)*k2 + (46732/5247)*k3 + (49/176)*k4 - (5013/18656)*k5,sigma,b,r,X,count)*dt
    k7 = func(t0 + dt,y0 + (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6,sigma,b,r,X,count)*dt
    
    yk = y0 + (35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6
    diff = min(abs((71/57600)*k1 - (71/166695)*k3 + (71/1920)*k4 - (17253/339200)*k5 + (22/525)*k6 - (1/40)*k7))
    #print(k1)
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

def bin_arr(n):
    
    bin = []
    for i in range(n):
        temp = random.randint(0, 1)
        bin.append(temp) 
        
    return np.array(bin)

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
    
    #
    #Lets get some sound data
    raw = wave.open('test.wav')
    n_samples = raw.getnframes()
    n_channels = raw.getnchannels()
    #print(n_channels)
    #print(n_samples)
    signal = raw.readframes(-1)
    signal = np.frombuffer(signal, dtype ="int16") 
    l_channel = signal[0::2]
    r_channel = signal[1::2]
    l_channel = l_channel
    f_rate = raw.getframerate()
    time = np.linspace(
        0, # start
        n_samples / f_rate,
        num = n_samples
    )
    # print(len(time))
    # print(f_rate)
    # print(n_samples/f_rate)
    # print(n_samples)
    # print(len(signal))
 

    fig = plt.figure('sound')
    plt.title("Sound Wave")
    plt.xlabel("Time")
    plt.plot(time, l_channel)
    plt.show()
    
    
    #
    #We now create the 'sending' part of this code
    
    saved_vals_1 = []
    saved_t_1 = []
    
    #Create initial conditions
    y0 = np.array([init_x, init_y, init_z])
    dt = 0.01
    
    #create a starting yk value
    yk = y0
    
    #What we want to do here is use our adaptive stepsize integral to get to roughly some defined time -> use a while loop
    t = 0
    dt = 1/f_rate
    #dt = 0.01
    
    #n_samples = 10000
    
    
    
    while len(saved_vals_1) <= n_samples-1:
    #add our values to our future arrays
        saved_vals_1.append(yk)
        saved_t_1.append(t)
                     
        t += dt
        yk = dor_p(sender, t, yk, sig, b_lor, r_lor, tol, dt)
    
    saved_vals_1 = np.array(saved_vals_1)
    saved_t_1 = np.array(saved_t_1)
    
    
    x = saved_vals_1[:,0]
    y = saved_vals_1[:,1]
    z = saved_vals_1[:,2]
    print('Part 1 done')
    #print(saved_t_1)
    
    
    
    
    #
    #Now we create our perturbed X
    steps = len(x)
    #s = bin_arr(n_samples)
    #s=0
    s = l_channel
    X = x + s
    
    
    #
    #Now that we have our perturbed X, time to re-run the system
    
    returned_vals = []
    returned_t = []
    t_new = 0
    
    yk = np.array([-10,-10,3])
    
    count = 0
    while len(returned_vals) <= n_samples-1:
        returned_vals.append(yk)
        returned_t.append(t_new)
                     
        t_new += dt
        yk = dor_p_rec(reciever, t_new, yk, sig, b_lor, r_lor, tol, dt, X, count)
        count += 1
        
    
    returned_vals = np.array(returned_vals)
    returned_t = np.array(returned_t)
    
    u = returned_vals[:,0]
    v = returned_vals[:,1] 
    w = returned_vals[:,2] 
    # print(s[steps-20:steps])
    # print(X[steps-20:steps]-u[steps-20:steps])
    
    recieved = (X-u)
    encrypted = X
    
    print(s)
    print(X)
    print(X-u)
    
    fig = plt.figure('testing')
    plt.plot(saved_t_1, s, label = 's')
    # plt.plot(saved_t_1, X-u, label = 'X-u')
    # plt.plot(saved_t_1, x, label = 'x')
    plt.legend()
    
    fig = plt.figure('Standard View', figsize = (10,10), frameon = False)
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')

    # plotting
    #you can comment out different ax.plot things if you only need certain parts
    ax.plot(x, y, z, 'b')
    ax.plot(u,v,w,'y')
    ax.set_title('Reciever')
    ax.axes.set_xlim3d(left = min(x), right = max(x))
    ax.axes.set_ylim3d(bottom = min(y), top = max(y))
    ax.axes.set_zlim3d(bottom = min(z), top = max(z))
    ax.set_facecolor('white')
    ax._axis3don = False
    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.zaxis.set_ticklabels([])
    ax.legend()
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.set_title('Original')
    ax1.plot(saved_t_1, s, color = 'b', label = 'Original')
    ax2.set_title('Encrypted')
    ax2.plot(saved_t_1, encrypted, color = 'g', label = 'Encrypted')
    ax3.set_title('Recieved')
    ax3.plot(saved_t_1, recieved, color = 'r', label = 'Recieved')
    plt.show()
    
    # print(f_rate)
    # print(len(recieved))
    
    write('Encrypted.wav', f_rate, encrypted.astype(np.int16))
    write('Output.wav', f_rate, recieved.astype(np.int16))
        
    
    # for r in range(1,251):
    #     saved_vals_1 = []
    #     saved_t_1 = []
        
    #     #Create initial conditions
    #     y0 = np.array([init_x, init_y, init_z])
    #     dt = 0.01
        
    #     #create a starting yk value
    #     yk = y0
        
    #     #What we want to do here is use our adaptive stepsize integral to get to roughly some defined time -> use a while loop
    #     t = 0
    #     dt = 0.001
        
        
        
    #     while t <= 100:
    #     #add our values to our future arrays
    #         saved_vals_1.append(yk)
    #         saved_t_1.append(t)
                         
    #         t += dt
    #         yk = dor_p(sender, t, yk, sig, b_lor, r, tol, dt)
        
    #     saved_vals_1 = np.array(saved_vals_1)
    #     saved_t_1 = np.array(saved_t_1)
        
        
    #     x = saved_vals_1[:,0]
    #     y = saved_vals_1[:,1]
    #     z = saved_vals_1[:,2]
        
    #     #
    #     #Now we create our perturbed X
    #     s = 0
    #     X = x + s
        
    #     #
    #     #Now that we have our perturbed X, time to re-run the system
        
    #     returned_vals = []
    #     returned_t = []
    #     t_new = 0
        
    #     count = 0
    #     while t_new <= 100:
    #         returned_vals.append(yk)
    #         returned_t.append(t_new)
                         
    #         t_new += dt
    #         yk = dor_p_rec(reciever, t_new, yk, sig, b_lor, r, tol, dt, X, count)
    #         count += 1
            
        
    #     returned_vals = np.array(returned_vals)
    #     returned_t = np.array(returned_t)
        
    #     u = returned_vals[:,0]
    #     v = returned_vals[:,1] 
    #     w = returned_vals[:,2] 
        
    #     steps = len(x)
    #     print('Step', r)
        
    #     if abs(max(u[steps-20:steps] - x[steps-20:steps])) < tol:
    #         print(abs(max(u[steps-20:steps] - x[steps-20:steps])))  
    #         if abs(max(v[steps-20:steps] - y[steps-20:steps])) < tol:
    #             if abs(max(w[steps-20:steps] - z[steps-20:steps])) < tol:
    #                 print('We want the value: ', r)
        
    
    
    
    return

main()


