
import matplotlib.pyplot as plt
from erddapClient import ERDDAP_Griddap
from netCDF4 import Dataset
from matplotlib import colormaps
from pathlib import Path
import numpy as np
from pycurrents.file.binfile_n import BinfileSet
import xarray as xr
import matplotlib.animation as animation

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
file_id = Dataset('/home/jamie/projects/rogerrevelle/data/noaacwBLENDEDsstDNDaily_e5b2_b4c7_9276_U1717797277907.nc')
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

waypoints = np.column_stack(([-74.3666, -71.5, -71.0, -70.67], [36.2333, 39.5, 39.5, 41.527]))
waypoints = np.array(waypoints)

sea = readrbins(pth ='/mnt/revelle-data/RR2407/adcp_uhdas/RR2407/rbin/', sensor = 'seapath380', tag = 'gps')
sea = sea[0::500]
x = sea[:, 2]
y = sea[:, 3]
s = sea.shape

tmp = np.empty((5,2,))
tmp[:] = np.nan

blur = np.linspace(0,1, 4)

fig, (ax1) = plt.subplots(1, 1, figsize=(15, 10))
ax1.contourf(ras.y, ras.x, ras[:, :], 100, cmap = "coolwarm")
ax1.contourf(mask.y, mask.x, mask[:,:], 1, colors = "black")
# Once I have position I should be fine with heading from the gyro. 
ax1.grid(color = "grey", linestyle = '--', alpha = 0.6)# visible=None)
c = ax1.contourf(ras.y, ras.x, ras[:, :], 100, cmap = "coolwarm")
cbar = fig.colorbar(c)
scat = ax1.scatter(x[0], y[0], s = 100, alpha = blur, color = "black", label='Roger Revelle')
# ax1.scatter(prev_pos[:,2], prev_pos[:,3], marker = ',', color = "black", s = 0.5, alpha = 0.5)
pion1 = ax1.scatter(tmp[:,0], tmp[:,1], color = "grey", marker = 'X', s = 100, label='Pioneer Pie')
pion2 = ax1.scatter(tmp[:,0], tmp[:,1], color = "grey", marker = 'X', s = 100, label='Pioneer Pie')

swot1 = ax1.scatter(swot_pos[0,0], swot_pos[0,1], color = "black", marker = 'X', s = 100, label='SWOT Pie')
swot2 = ax1.scatter(swot_pos[1,0], swot_pos[1,1], color = "black", marker = 'X', s = 100, label='SWOT Pie')
swot3 = ax1.scatter(swot_pos[2,0], swot_pos[2,1], color = "black", marker = 'X', s = 100, label='SWOT Pie')

ax1.scatter(waypoints[:,0], waypoints[:,1], color = "red", marker = '^', s = 100, label='Waypoints')
# cbar.set_label("Sea Surface Temperature [C$^\circ$]")
ax1.set(xlim=[-77.5, -64], ylim=[34, 44], xlabel='Longitude', ylabel='Latitude')
ax1.set_title("2024-06-07 Sea Surface Temperature [C$^\circ$]", size = 15)
ax1.legend(loc = 'upper right')

def update(frame):
    # for each frame, update the data stored on each artist.
    u = x[(frame-4):frame]
    v = y[(frame-4):frame]
    # update the scatter plot:
    data = np.stack([u, v]).T
    scat.set_offsets(data)
    
    # Deployed
    if frame > 100:
        w = pioneer_pos[0,0]
        z = pioneer_pos[0,1]
        # update the scatter plot:
        data = np.stack([w, z]).T
        pion1.set_offsets(data)

    if frame > 80:
        w = pioneer_pos[1,0]
        z = pioneer_pos[1,1]
        # update the scatter plot:
        data = np.stack([w, z]).T
        pion2.set_offsets(data)

    # Retreaved
    if frame > 100:
            w = tmp[0,0]
            z = tmp[0,0]
            # update the scatter plot:
            data = np.stack([w, z]).T
            swot1.set_offsets(data)
    
    if frame > 110:
        w = tmp[0,0]
        z = tmp[0,0]
        # update the scatter plot:
        data2 = np.stack([w, z]).T
        swot2.set_offsets(data2)

    if frame > 120:
        w = tmp[0,0]
        z = tmp[0,0]
        # update the scatter plot:
        data3 = np.stack([w, z]).T
        swot3.set_offsets(data3)

    # update the line plot:
    #return(scat)
ani = animation.FuncAnimation(fig=fig, func=update, frames=s[0], interval=0.1)
plt.show()
