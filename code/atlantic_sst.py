import matplotlib.pyplot as plt
from erddapClient import ERDDAP_Griddap
from netCDF4 import Dataset
from matplotlib import colormaps
from pathlib import Path
import numpy as np
from pycurrents.file.binfile_n import BinfileSet
import xarray as xr

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

# Sea-Surface Temperature, NOAA Geo-polar Blended Analysis Day+Night, GHRSST, Near Real-Time, Global 5km, 2019-Present, Daily 
file_id = Dataset('/home/jamie/projects/rogerrevelle/data/noaacwBLENDEDsstDNDaily_815e_004d_6cff_U1718113745770.nc') 

# get gps location
sea = readrbins(pth ='/mnt/revelle-data/RR2407/adcp_uhdas/RR2407/rbin/', sensor = 'seapath380', tag = 'gps')
gyro = readrbins(pth ='/mnt/revelle-data/RR2407/adcp_uhdas/RR2407/rbin/', sensor = 'gyro', tag = 'hdg')
# pull variables from nc file. 
ras = file_id.variables["analysed_sst"][:]
lat = file_id.variables["latitude"][:]
lon = file_id.variables["longitude"][:]
mask = file_id.variables["mask"][:]
file_id.close()

# convert to xarray. 
ras = xr.DataArray(ras[0,:,:], 
                       coords={'x': lat, 'y':lon}, 
                       dims=["x", "y"])

mask = xr.DataArray(mask[0,:,:], 
                       coords={'x': lat, 'y':lon}, 
                       dims=["x", "y"])
mask = mask.where(mask.values != 1)

# PIES locations
swot_pos = np.column_stack(([-74.3666, -74.5339, -74.1932, -74.6728, -74.3657, -74.2781], 
                           [36.2333, 35.9997, 36.0069, 36.0001, 36.0012, 35.7561]))
swot_pos = np.array(swot_pos)
pioneer_pos = np.column_stack(([-74.705, -74.7633], [36.050, 35.700]))
pioneer_pos = np.array(pioneer_pos)

#waypoints = np.column_stack(([-74.3666, -71.5, -71.0, -70.67], [36.2333, 39.5, 39.5, 41.527]))
#waypoints = np.array(waypoints)

# pull out most regent heading and convert to radians. 
theta = gyro[-1,1] *(np.pi/180) # to radians
# pos = sea[-1]
# grab last 10 positions. 
pos = sea[0::1500]

fig, (ax1) = plt.subplots(1, 1, figsize=(15, 10))
ax1.contourf(ras.y, ras.x, ras[:, :], 100, cmap = "coolwarm")
ax1.contourf(mask.y, mask.x, mask[:,:], 1, colors = "black")
# Once I have position I should be fine with heading from the gyro. 
ax1.grid(color = "grey", linestyle = '--', alpha = 0.6)# visible=None)
c = ax1.contourf(ras.y, ras.x, ras[:, :], 100, cmap = "coolwarm")
cbar = fig.colorbar(c)
ax1.scatter(pos[:,2], pos[:,3], alpha = 0.8, s = 2, color = "black", label='Cruise Track')
# ax1.scatter(prev_pos[:,2], prev_pos[:,3], marker = ',', color = "black", s = 0.5, alpha = 0.5)
ax1.scatter(pioneer_pos[:,0], pioneer_pos[:,1], color = "grey", marker = 'X', s = 100, label='Pioneer Pie')
ax1.scatter(swot_pos[:,0], swot_pos[:,1], color = "black", marker = 'X', s = 100, label='SWOT Pie')
# ax1.scatter(pos[:,2], pos[:,3], alpha = 0.8, s = 2, color = "black", label='Cruise Track')
# ax1.scatter(waypoints[:,0], waypoints[:,1], color = "red", marker = '^', s = 100, label='Waypoints')
ax1.set_xlim(-77.5, -66) #22
ax1.set_ylim(34, 43) #16
# cbar.set_label("Sea Surface Temperature [C$^\circ$]")
ax1.set_xlabel("Longitude [$^\circ W$]", size = 11)
ax1.set_ylabel("Latitude [$^\circ N$]", size = 11)
ax1.set_title("2024-06-09 Sea Surface Temperature [C$^\circ$]", size = 15)
ax1.legend(loc = 'upper right')
plt.savefig('../figures/atlantic_sst.pdf', dpi=300)
plt.show();