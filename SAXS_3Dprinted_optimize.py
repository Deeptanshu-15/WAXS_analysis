# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:08:42 2021

@author: sid
"""
from matplotlib import pyplot as plt
import numpy as np
import pyFAI, pyFAI.detectors, fabio
import pyFAI.distortion as dis
from pyFAI.gui import jupyter
from scipy import ndimage
from scipy import optimize as op
from scipy.fft import fft, fftfreq
from pyFAI.calibrant import get_calibrant
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import time
import lmfit
import math

print("Using pyFAI version", pyFAI.version)
plt.ion()
#Detector calibration and setup

#openfile for data #Only RAW DATA has radial averaging, as urwarped is already averaged
fig_Xfil = fabio.open(r"C:\PhD work\PhD_Nov21\Sample14_Mono_30s_107cm_01_unwarped.gfrm") #data has additional commands like shape
img_Xfil = fig_Xfil.data
Mask_dat = fabio.open(r"C:\PhD work\PhD_Nov21\Emptybeam_107cm_01_unwarped.gfrm") #Air or empty beam measurement as mask
msk = Mask_dat.data #Mask_correction

#for headers -> fig_z.header()
#Detector and measurement parameters
wl = 1.54184e-10 #nm dimension of X-rays
cal = get_calibrant("AgBh") #Silver behanate sample
cal.wavelength=wl
start_time = time.time()
print("PyFAI version", pyFAI.version)
Vantec = pyFAI.detectors.Detector(68e-6, 68e-6)#pixel size
Vantec.max_shape=(2048,2048)#image shape
ai = AzimuthalIntegrator(dist=1.07050, detector=Vantec, wavelength=wl)#initialization of arbitrary detector with given dimensions
#Masks and darks
Ai_mask = ai.create_mask(msk)
ai.mask = Ai_mask
#image center calculation
cent = img_Xfil.T
x_cent = np.zeros(len(cent))
x_holder = np.zeros(len(cent))
y_cent = np.zeros(len(cent))
y_holder = np.zeros(len(cent))
#image center calculation
for i in range(len(cent)):
    for j in range(len(cent)):
        x_holder[j] = cent[i][j] #running X center intensity loop
        y_holder[j] = cent[j][i] #running Y center intensity loop
    x_cent[i] = x_holder.sum()
    y_cent[i] = y_holder.sum()
x_c=y_c = 0
for i in range(len(cent)):
    ctr_x = (x_cent[i]*i)
    ctr_y = (y_cent[i]*i)
    x_c+=ctr_x
    y_c+=ctr_y

xx_ctr = x_c/x_cent.sum()
yy_ctr = y_c/y_cent.sum() #weighted average for center position

#finding beamcenter with PONI=Point of normal incedence
p1 = 68e-6 * 2048/2
ai.poni1 = p1 - 0.00017
p2 = 68e-6 * 2048/2
ai.poni2 = p2
print(ai)
fake = cal.fake_calibration_image(ai)

distort = dis.Distortion(detector=Vantec, shape=Vantec.shape, resize = False, empty=0,mask=msk,method = 'lut')
cor_img = distort.correct_ng(img_Xfil, solidangle = ai.solidAngleArray)
plt.rcParams['figure.dpi'] = 600 #inline plot dpi setting
plt.rcParams["figure.figsize"] = (16,9)

ai.setSPD(SampleDistance = 1.070500031, Center_1=xx_ctr, Center_2 = yy_ctr) #maybe removed
Desc = 1500 #Descretizer for 2D unwrapping of the scattering
#Plot 1D azimuthal and radial integration--was not removed before; but results are varying. better to use for 1d integrate. radial integration is questionable
d_z = ai.integrate2d(img_Xfil, Desc, correctSolidAngle = False, radial_range = [0.3,1.0])

#plotting 2d integrated data
intensity, q, tth = d_z
z_sum = np.zeros(len(intensity))
                 
for i in range(len(intensity)): ## Z-axis aligned sample distortion correction for initial beamstop
    z_sum[i] = intensity[i].sum()
    
#moving averaging for correction    
def moving_average(x): #(x,w)
    mv = np.zeros(len(intensity))
    for i in range(len(mv)):
        mv[i] = (x[i-2] + x[i-1] +x[i])/3
    return mv

def peak_opt(xx_ctr, yy_ctr):
    store=0
    ai.setSPD(SampleDistance = 1.070500031, Center_1=xx_ctr, Center_2 = yy_ctr) #maybe removed
    Desc = 1500 #Descretizer for 2D unwrapping of the scattering
    d_z = ai.integrate2d(img_Xfil, Desc, correctSolidAngle = False, radial_range = [0.3,1.0])
    intensity, q, tth = d_z
    z_sum = np.zeros(len(intensity))
    for i in range(len(intensity)): ## Z-axis aligned sample distortion correction for initial beamstop
        z_sum[i] = intensity[i].sum()
    error = abs(z_sum[0:150].max() - z_sum[170:300].max()) #peak_max difference between 180 peak and 0 peak; difference is conditional factor
    if (error<((0.002*z_sum.max()))): #tightening factor = better fit is preferred
        err_con = 1
        store = error
        print (error, "convergence")
    else:
        store = error
        err_con = 0
    return err_con, store

# counter = 0
# for i in range(-5,5,1):
#     for j in range(-5,5,1):
#         y_ctr = yy_ctr
#         a = peak_opt(xx_ctr,yy_ctr+j)
#         if(a==1):
#             y_ctr = yy_ctr+j
#             counter +=1
#             print(y_ctr)
#             # print("y_ctr found {}".format(j))
#             #break
#     b = peak_opt(xx_ctr+i, y_ctr)
#     if(b==1):
#         x_ctr = xx_ctr+i
#         counter+=1
#         print(x_ctr, y_ctr)

counter = 0
conv = peak_opt(xx_ctr, yy_ctr)[1]
x_ctr=0
y_ctr=0
for i in np.arange(-3,3,0.5):
    for j in np.arange(-3,3,0.5):
        a, b = peak_opt(xx_ctr+i,yy_ctr+j)
        if(a==1 and b<=conv):
            conv=b
            x_ctr = xx_ctr+i
            y_ctr = yy_ctr+j
            print(x_ctr, y_ctr)
    if(b<20):
          print("A convergence")
          break  
            
ai.setSPD(SampleDistance = 1.070500031, Center_1=x_ctr, Center_2 = y_ctr) #maybe removed
Desc = 500 #Descretizer for 2D unwrapping of the scattering
#Plot 1D azimuthal and radial integration--was not removed before; but results are varying. better to use for 1d integrate. radial integration is questionable
d_z = ai.integrate2d(img_Xfil, Desc, correctSolidAngle = False, radial_range = [0.3,1.0])

#plotting 2d integrated data
intensity, q, tth = d_z
z_sum = np.zeros(len(intensity))
                 
for i in range(len(intensity)): ## Z-axis aligned sample distortion correction for initial beamstop
    z_sum[i] = intensity[i].sum()
index_z = np.argmin(z_sum)
rotate_img_z = ndimage.rotate(img_Xfil,0, reshape=False) #rotation to index of minimum theta where we can shift/rotate image
proc_img_z = rotate_img_z[800:1248, 800:1248]

fig,ax = plt.subplots()
plt.plot(tth, z_sum)
plt.xlabel('Azimuthal angle, $\mathregular {^{°}}$', labelpad=20, fontsize=32)
plt.ylabel('Intensity, a.U.', labelpad=20, fontsize = 32)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.figtext(.7, .7, "X_fil", fontsize = 24)
plt.xlim([-180,180])
plt.ylim([0,z_sum.max()+1000])
plt.tick_params(axis='both', pad = 10, top=True, right=True)
plt.grid(which='major', color='#DDDDDD', linewidth=1.25)
plt.minorticks_on()
for axis in ['top', 'bottom', 'left','right']:
    ax.spines[axis].set_linewidth(2.0)
plt.show()

import matplotlib.patches as patches
plt.figure(figsize=(16,9))
plt.axis("off")
z = plt.imshow(proc_img_z, cmap = 'plasma', vmin=0, vmax = 500)
plt.colorbar() # Show color bar of above image
plt.show()

def area_DPO(a): # REAL _ DPO calculation basically area under peak/area under entire curve
    h = a.min()
    holder=[]
    denom = a.sum()
    numer = h*360
    return (denom-numer)/denom

def hermann(a, tth):
    num = np.zeros(360)
    dem = np.zeros(360)
    for i in range(360):
            num[i] = a[i]*np.cos(np.deg2rad(tth[i]))**2*np.sin(np.deg2rad(tth[i]))
            dem[i] = a[i]*np.sin(np.deg2rad(tth[i]))
    cos = num.sum()/dem.sum()
    num2=np.absolute(num)
    dem2=np.absolute(dem)
    cos2 = num2.sum()/dem2.sum()
    gamma = 1-(2*cos2)
    f=(3*gamma-1)/2
    return f

# I (n2)  a0 + a1 cos(2the*n**2/N**2 −2s)
# aa, bb = op.curve_fit(lambda t, a, b,c,d: a + b*np.cos(2*np.pi*t/c+d), tth, z_sum)
# def func(x,a,b,c,d):
#     return a + b*np.cos(2*np.pi*x/c+d)
# # plt.plot(tth, *aa)

# yf = fft(20340 + tth*np.sin(tth/57 - 34.778))

def lmfit_Int(test, test1, peak1, tth1):
    gauss1 = lmfit.models.LorentzianModel(prefix='g1_')
    pars=gauss1.guess(test, x=test1)
    pars['g1_center'].set(value=-135, min=-180, max=180)
    pars['g1_sigma'].set(value=200, min=-100)
    pars['g1_amplitude'].set(value=100, min=1)
    init = gauss1.eval(pars, x=test1)
    out = gauss1.fit(test, pars, x=test1)
    fwhm = 2*out.best_values['g1_sigma']*np.sqrt(2*np.log(2))
    gauss2 = lmfit.models.LorentzianModel(prefix='g2_')
    pars2=gauss2.guess(peak1, x=tth1)
    pars2['g2_center'].set(value=50, min=-180, max=180)
    pars2['g2_sigma'].set(value=200, min=-100)
    pars2['g2_amplitude'].set(value=100, min=1)
    init_2 = gauss2.eval(pars2, x=tth1)
    out_2 = gauss2.fit(peak1, pars2, x=tth1)
    fwhm2 = 2*out_2.best_values['g2_sigma']*np.sqrt(2*np.log(2))
    pie = (180-fwhm)/180
    # print(out.fit_report(min_correl=0.01))
    return out.best_fit, out_2.best_fit, fwhm,fwhm2

DPO = area_DPO(z_sum)
Hermann = hermann(z_sum, tth)
print ("DPO: %f \t Herman's parameter: %f"%(DPO, Hermann))

ctr=0
ct=0
c=0
ctri=np.zeros(25)
HM_1= np.zeros(25)
HM_2 = np.zeros(25)
hold_q=np.zeros(25)
start = 0.5 #0.25--0.1, step 0.07
for i in range(Desc):
    if (q[i]>start and q[i]<1.0):
        ctri[ct]=i
        hold_q[ct] = q[i]
        start+=0.03
        ct+=1
       # break
Icum=np.zeros((len(tth),ct))
#     plt.plot(tth[170:290], Imat[170:290:,int(ctri[j])])
for i in range(ct):
    for j in range(len(tth)):
        Icum[j][c]+=intensity[j][int(ctri[c])]
    c+=1    
# ct=ct-1

n=np.zeros((len(tth), ct))
nn =np.zeros((len(tth), ct))
spare = np.zeros((len(tth), ct))
tth2 = np.zeros(len(tth))
clr=[]
clr_z=[]
for i in range(len(tth)):
    hol=0
    for j in range(ct):
        hol += Icum[i][j]
        spare[i][j]=hol  #actual cumulative

for i in range(len(tth)):
    for j in range(ct):    
        nn[i][j] = spare[i-20][j]
    tth2[i] = tth[i]
    
for i in range(ct):
    nn[0:180,i] = nn[0:180,i]-nn[0:180,i].min()
    nn[180:360,i] = nn[180:360,i]-nn[180:360,i].min()
    clr.append('b')
    clr_z.append('black')  

fig, ax = plt.subplots()
x=10
for i in range(ct):
    plt.scatter(tth2[0:160]-x, nn[0:160,i], color=clr_z[i],s=15, marker= 'x') #120° profile for lorentzian fit
    plt.scatter(tth2[160:320]-x, nn[160:320,i], color=clr_z[i],s=15, marker= 'o')
    # plt.plot(tth[0:170], np.hstack((lmfit_Int(n[0:65,i], tth[0:65], n[65:170,i], tth[65:170])[0], lmfit_Int(n[0:65,i], tth[0:65], n[65:170,i], tth[65:170])[1])))
    plt.plot(tth2[0:140]-x, np.hstack((lmfit_Int(nn[0:140,i], tth2[0:140]-x, nn[160:300,i], tth2[160:300])[0])), color=clr[i])
    plt.plot(tth2[160:300]-x, np.hstack((lmfit_Int(nn[0:140,i], tth2[0:140]-x, nn[160:300,i], tth2[160:300])[1])),color=clr[i])
    HM_1[i] = lmfit_Int(nn[0:140,i], tth2[0:140]-x, nn[160:300,i], tth2[160:300])[2]
    HM_2[i] = lmfit_Int(nn[0:140,i], tth2[0:140]-x, nn[160:300,i], tth2[160:300])[3]
    plt.xlim([-200, 180])
    plt.ylim([0, 100])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tick_params(axis='both', pad = 5, top=True, right=True, length=10, width = 1.5)
    plt.title('XZ_Concentrated', fontsize=24)
    plt.rcParams['axes.titley'] = 0.9  

for i in range(ct):
    HM_1[i] = math.radians(HM_1[i])
    HM_2[i] = math.radians(HM_2[i])
    # hold_q[i] =10*hold_q[i]
plt.rcParams["figure.figsize"] = (10,10)
lin = lmfit.models.LinearModel()
out1 = lin.fit(HM_1[10:15], x=1/hold_q[10:15])
out2 = lin.fit(HM_2[3:15], x=1/hold_q[3:15])
# out_holder = np.zeros(len(out2.best_fit))
Lf = 1/out2.best_values['slope']
Lf_1 = 1/out1.best_values['slope']
cept = math.degrees(out2.best_values['intercept'])
cept_1 = math.degrees(out1.best_values['intercept'])

rld = np.zeros(4)
rld[0] = cept
rld[1] = cept_1
rld[2] = Lf*10
rld[3] = Lf_1*10
aa, bb = op.curve_fit(lambda t, a, b,d: a + b*np.cos(2*np.pi*t/360-d), tth, z_sum)
def func(x,a,b,d):
    return a + b*np.cos(2*np.pi*x/360-d)
# plt.plot(tth, *aa)

Four_parm = aa[1]/aa[0]

print ("Bphi: %f ; Bphi_2: %f, Fibre length:%f %f" %(cept, cept_1, Lf, Lf_1))

# fig, ax = plt.subplots()
# plt.scatter(1/hold_q, holder,s=150)
# plt.scatter(1/hold_q, holder_1, s=150)
# plt.plot(1/hold_q[0:5], out2.best_fit) #out2.best_fit
# plt.plot(1/hold_q[0:5], out1.best_fit)
# plt.xlabel('1/q (scattering vector, $\mathregular {nm^{-1}}$)', labelpad=20, fontsize=32)
# # plt.legend(fontsize=22, frameon=False, loc='best', bbox_to_anchor=(0.6, 0.4))
# plt.ylabel('$\mathregular {B_{obs}}$(radians)', labelpad=20, fontsize = 32)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.figtext(.7, .7, "m = 0.048\n $\mathregular {R^{2}}$=0.98", fontsize = 24)
# plt.xlim([0,4])
# plt.ylim([0,3])
# plt.tick_params(axis='both', pad = 10, top=True, right=True)
# plt.grid(which='major', color='#DDDDDD', linewidth=1.25)
# plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=1.0)
# plt.minorticks_on()
# for axis in ['top', 'bottom', 'left','right']:
#     ax.spines[axis].set_linewidth(2.0)
# plt.show()

# I (n2)  a0 + a1 cos(2the*n**2/N**2 −2s)
aa, bb = op.curve_fit(lambda t, a, b,d: a + b*np.cos(2*np.pi*t/0.8-d), tth[0:180], z_sum[0:180])
def func(x,a,b,d):
    return a + b*np.cos(2*np.pi*x/0.8-d)

Four_parm = aa[1]/aa[0]
print("FT_fit: %f"%(Four_parm))