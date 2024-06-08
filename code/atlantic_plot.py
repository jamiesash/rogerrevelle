import matplotlib.pyplot as plt
from erddapClient import ERDDAP_Griddap
from netCDF4 import Dataset
from matplotlib import colormaps
from pathlib import Path
import numpy as np
from pycurrents.file.binfile_n import BinfileSet

def arrayrbins(files):
    mat = list()
    for i in files:
        tmp = BinfileSet(str(i))
        mat.append(tmp.array)
    mat = np.vstack(mat)
    return(mat) 

def readrbins(pth, sensor, tag):
    tag = "*" + tag + "*.rbin"
    files = sorted(Path(pth+sensor+"/").glob(tag))
    mat = arrayrbins(files)

    #cols = BinfileSet(str(files[0])).columns
    mat = np.array(mat) #, dtype=cols)
    return(mat)

# Read in the data. 
file_id = Dataset('/home/jamie/projects/atlantic_sst/noaacwBLENDEDsstDNDaily_e5b2_b4c7_9276_U1717797277907.nc')

# get gps location
sea = readrbins(pth ='/home/jamie/projects/atlantic_sst/rbin/', sensor = 'seapath380', tag = 'gps')
gyro = readrbins(pth ='/home/jamie/projects/atlantic_sst/rbin/', sensor = 'gyro', tag = 'hdg')
# pull variables from nc file. 
sst = file_id.variables["analysed_sst"][:]
lat = file_id.variables["latitude"][:]
lon = file_id.variables["longitude"][:]
mask = file_id.variables["mask"][:]
file_id.close()

#theta = 360 *(np.pi/180) # to radians
# pull out most regent heading and convert to radians. 
theta = gyro[-1,1] *(np.pi/180) # to radians
pos = sea[-1]

fig, (ax1) = plt.subplots(1, 1, figsize=(15, 10))
ax1.contourf(lon, lat, mask[0,:,:], cmap = "binary")
ax1.contourf(lon, lat, sst[0,:,:], 100, cmap = "coolwarm")
# Once I have position I should be fine with heading from the gyro. 
ax1.grid(color = "grey", linestyle = '--', alpha = 0.6)# visible=None)
c = ax1.contourf(lon, lat, sst[0,:,:], 100, cmap = "coolwarm")
cbar = fig.colorbar(c)
ax1.quiver(pos[2], pos[3], np.cos(theta), np.sin(theta), headlength=0.0001, headaxislength=0.0001, width = 0.005)
ax1.scatter(pos[2], pos[3], color = "black")
# cbar.set_label("Sea Surface Temperature [C$^\circ$]")
ax1.set_xlabel("Longitude [$^\circ W$]", size = 11)
ax1.set_ylabel("Latitude [$^\circ N$]", size = 11)
ax1.set_title("2024-06-06 Sea Surface Temperature [C$^\circ$] and Ship's Current Position", size = 11);
plt.show()