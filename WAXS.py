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

print("Using pyFAI version", pyFAI.version)
plt.ion()
#Detector calibration and setup

#openfile for data #Only RAW DATA has radial averaging, as urwarped is already averaged
fig_Xfil = fabio.open(r"C:\Users\sid\Desktop\3D printed CNF_CNC_plus\SAXS 499_24_03\WAXS\s24_60s_5cm_01_unwarped.gfrm") #data has additional commands like shape
img_Xfil = fig_Xfil.data
Mask_dat = fabio.open(r"C:\Users\sid\Desktop\3D printed CNF_CNC_plus\SAXS 499_24_03\WAXS\empty_60s_5cm_01_unwarped.gfrm") #Air or empty beam measurement as mask
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
ai = AzimuthalIntegrator(dist=0.0505, detector=Vantec, wavelength=wl)#initialization of arbitrary detector with given dimensions
#Masks and darks
Ai_mask = ai.create_mask(msk)
ai.mask = Ai_mask
#image center calculation
x_ctr = img_Xfil.sum(axis=0).argmax()
y_ctr = img_Xfil.sum(axis=1).argmax()

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

ai.setSPD(SampleDistance = 0.0505, Center_1=x_ctr, Center_2 = y_ctr) #maybe removed
Desc = 1500 #Descretizer for 2D unwrapping of the scattering
#Plot 1D azimuthal and radial integration--was not removed before; but results are varying. better to use for 1d integrate. radial integration is questionable
d_z = ai.integrate2d(img_Xfil, Desc, correctSolidAngle = False, radial_range = [3.0,30.0])
# jupyter.display(img_Xfil)
# jupyter.plot2d(d_z, label = 'S20')
#plotting 2d integrated data
intensity, q, tth = d_z
z_sum = np.zeros(len(intensity))
for i in range(len(intensity)): ## Z-axis aligned sample distortion correction for initial beamstop
    z_sum[i] = intensity[i].sum()
# plt.plot(tth,z_sum)
#moving averaging for correction    
def moving_average(x): #(x,w)
    mv = np.zeros(len(intensity))
    for i in range(len(mv)):
        mv[i] = (x[i-2] + x[i-1] +x[i])/3
    return mv

mv_avg = moving_average(z_sum)
# fig_avg = px.scatter(x= tth, y = mv_avg, height = 1200, width = 1200,labels = 'Moving average')
error = (z_sum.max() + z_sum.min())/2
err= np.zeros(len(intensity))
for i in range(360):
    err[i] = abs(np.average(z_sum)-z_sum[i])
print (err.sum(), np.average(z_sum))

index_z = np.argmin(z_sum)
rotate_img_z = ndimage.rotate(img_Xfil,0, reshape=False) #rotation to index of minimum theta where we can shift/rotate image
proc_img_z = rotate_img_z[800:1248, 800:1248]

#correction and normalization protocol
z_norm = np.zeros(20)
z_norm = z_sum[z_sum.argsort()][:20]
# rot_z = integrate2d(rotate_img_z, Desc, radialrange=[0.2, 2.5], mask=msk)
fig, ax = plt.subplots()

poly_z = np.polyfit(tth, z_sum-z_norm.mean(),8)
poly_yy = np.poly1d(poly_z)(tth)
plt.plot(tth, poly_yy)
plt.xlabel('Azimuthal angle, $\mathregular {^{°}}$', labelpad=20, fontsize=32)
plt.ylabel('Intensity, a.U.', labelpad=20, fontsize = 32)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.figtext(.7, .7, "X_fil", fontsize = 24)
plt.xlim([-180,180])
plt.ylim([0,7000])
plt.tick_params(axis='both', pad = 10, top=True, right=True)
plt.grid(which='major', color='#DDDDDD', linewidth=1.25)
plt.minorticks_on()
for axis in ['top', 'bottom', 'left','right']:
    ax.spines[axis].set_linewidth(2.0)
plt.show()
# plt.show()

#showing the 2D scatter plot better than jupyter
import matplotlib.patches as patches
import matplotlib
plt.figure(figsize=(16,9))
plt.axis("off")
z = plt.imshow(img_Xfil, cmap = 'plasma', norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=100))
plt.colorbar() # Show color bar of above image
plt.show()

def area_DPO(a): # REAL _ DPO calculation basically area under peak/area under entire curve
    h = a.min()
    holder=[]
    denom = a.sum()
    numer = h*360
    return (denom-numer)/denom

def hermann(a, tth):
    i = a[0:180].argmax()
    num = np.zeros(90)
    dem = np.zeros(90)
    for c in range(90):
            num[c] = a[c+i]*np.cos(tth[c+i])**2*np.sin(tth[c+i])
            dem[c] = a[c+i]*np.sin(tth[c+i])
    cos = num.sum()/dem.sum()
    gamma = 1-(2*cos)
    f=(3*gamma-1)/2
    return cos, f

def hermann2(a, tth):
    num = np.zeros(360)
    dem = np.zeros(360)
    for i in range(360):
            num[i] = a[i]*np.cos(tth[i])**2*np.sin(tth[i])
            dem[i] = a[i]*np.sin(tth[i])
    cos = num.sum()/dem.sum()
    gamma = 1-(2*cos)
    f=(3*cos-1)/2
    return cos, f

def S_fact(a, tth):
    num = np.zeros(90)
    for i in range(90):
        num[i] = a[i]*(3*np.cos(tth[i])**2-1)*np.sin(tth[i])
    S = np.pi*num.sum()
    return S

def Wf_fact(a, tth):
    num = np.zeros(90)
    for i in range(90):
        num[i] = np.pi*(a[i]*(3*np.cos(tth[i])**2-1)*np.sin(tth[i]))
    S = num.sum()
    return S

# def OP():
#     gauss1 = lmfit.models.LorentzianModel(prefix='g1_')
#     pars=gauss1.guess(test, x=test1)
#     pars['g1_center'].set(value=-135, min=-180, max=180)
#     pars['g1_sigma'].set(value=200, min=-100)
#     pars['g1_amplitude'].set(value=100, min=1)
#     init = gauss1.eval(pars, x=test1)
#     out = gauss1.fit(test, pars, x=test1)
#     fwhm = 2*out.best_values['g1_sigma']*np.sqrt(2*np.log(2))
#     num = np.zeros(360)
#     dem = np.zeros(360)
#     for i in range(360):
#             num[i] = a[i]*np.cos(np.deg2rad(tth[i]))**2*np.sin(np.deg2rad(tth[i]))
#             dem[i] = a[i]*np.sin(np.deg2rad(tth[i]))
#     cos = num.sum()/dem.sum()
#     num2=np.absolute(num)
#     dem2=np.absolute(dem)
#     cos2 = num2.sum()/dem2.sum()
#     gamma = 1-(2*cos2)
#     f=(3*gamma-1)/2
#     return f

tth2 = tth+180
DPO = area_DPO(z_sum)
Hermann = hermann(z_sum, tth2)
herm2 = hermann2(z_sum,tth2)
print ("DPO: %f \t Herman's: %f %f"%(DPO, Hermann[0], Hermann[1]))
print ("\t Herman's(360): %f %f"%(herm2[0], herm2[1]))
