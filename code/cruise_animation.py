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

def findframe(gps, point):
    A = gps[:,0] > point[0] - 0.01
    B = gps[:,0] < point[0] + 0.01 
    C = gps[:,1] > point[1] - 0.01
    D = gps[:,1] < point[1] + 0.01
    a = np.logical_and(A , B)
    b = np.logical_and(C , D)
    c = np.logical_and(a , b)
    idx = np.where(c)
    frame = np.min(idx)
    return(frame)

# ----------------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------------
### PIES locations
swot_pos = np.column_stack(([-74.3666, -74.5339, -74.1932, -74.6728, -74.3657, -74.2781], 
                           [36.2333, 35.9997, 36.0069, 36.0001, 36.0012, 35.7561]))
swot_pos = np.array(swot_pos)
pioneer_pos = np.column_stack(([-74.705, -74.7633], [36.050, 35.700]))
pioneer_pos = np.array(pioneer_pos)

waypoints = np.column_stack(([-74.3666, -71.5, -71.0, -70.67], [36.2333, 39.5, 39.5, 41.527]))
waypoints = np.array(waypoints)

# ----------------------------------------------------------------------------------------------------------------------------
### CTD locations. 

ctd = np.genfromtxt('../data/CTD_positions.csv', delimiter=',')

# ----------------------------------------------------------------------------------------------------------------------------
### SHIP GPS Positions
sea = readrbins(pth ='/mnt/revelle-data/RR2407/adcp_uhdas/RR2407/rbin/', sensor = 'seapath380', tag = 'gps')
sea = sea[0::500]
u0 = sea[:, 2]
v0 = sea[:, 3]
s = sea.shape
pos = np.array([u0, v0]).T

# ----------------------------------------------------------------------------------------------------------------------------
### Drifter Psotions. 

# drift1 = readrbins(pth ='/mnt/revelle-data/RR2407/adcp_uhdas/RR2407/rbin/', sensor = 'seapath380', tag = 'gps')

# Make time the same cardinality as ship position (already done if using seapath time stamp). 
# Find the time the drifter was deployed,
# Find the ships location at that time. 
# From that time going forwards, make the ship location and drifter location the same cardinality.  
# Find frame using location for ease of use. 

# ----------------------------------------------------------------------------------------------------------------------------
# Dumb stuff for plotting.

tmp = np.empty((5,2,))
tmp[:] = np.nan
blur = np.linspace(0,1, 5)

# ----------------------------------------------------------------------------------------------------------------------------
fig, (ax1) = plt.subplots(1, 1, figsize=(15, 10))
ax1.contourf(ras.y, ras.x, ras[:, :], 100, cmap = "coolwarm")
ax1.contourf(mask.y, mask.x, mask[:,:], 1, colors = "black")
# Once I have position I should be fine with heading from the gyro. 
ax1.grid(color = "grey", linestyle = '--', alpha = 0.6)# visible=None)
cm = ax1.contourf(ras.y, ras.x, ras[:, :], 100, cmap = "coolwarm")
cbar = fig.colorbar(cm)
scat = ax1.scatter(u0[0], v0[0], s = 100, alpha = blur, color = "black", label='Roger Revelle')

# Initillize the drifter plot. 
#drifter1 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, alpha = blur, color = "black", label='Drifters')
#drifter2 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, alpha = blur, color = "black")
#drifter3 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, alpha = blur, color = "black")

# Initillize the CTD plot. 
pion1 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = 'X', s = 100, label='Pioneer Pies')
#ctd2 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
#ctd3 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
##ctd4 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
#ctd5 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
#ctd6 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
#ctd7 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
#ctd8 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
#ctd9 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
#ctd10 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
#ctd11 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
#ctd12 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
#ctd13 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
#ctd14 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
#ctd15 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")
#ctd16 = ax1.scatter(tmp[0,0], tmp[0,1], s = 100, marker = 'o', color = "black")

# Deply instruments
pion1 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey", marker = 'X', s = 100, label='Pioneer Pies')
pion2 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey", marker = 'X', s = 100)

# Initilize the swot. 
swot1 = ax1.scatter(swot_pos[0,0], swot_pos[0,1], color = "black", marker = 'X', s = 100, label='SWOT Pies')
swot2 = ax1.scatter(swot_pos[1,0], swot_pos[1,1], color = "black", marker = 'X', s = 100)
swot3 = ax1.scatter(swot_pos[2,0], swot_pos[2,1], color = "black", marker = 'X', s = 100)
swot4 = ax1.scatter(swot_pos[3,0], swot_pos[3,1], color = "black", marker = 'X', s = 100)
swot5 = ax1.scatter(swot_pos[4,0], swot_pos[4,1], color = "black", marker = 'X', s = 100)
swot6 = ax1.scatter(swot_pos[5,0], swot_pos[5,1], color = "black", marker = 'X', s = 100)

ax1.scatter(ctd[:,0], ctd[:,1], color = "black", marker = 'o', s = 100, label='CTD')
ax1.scatter(waypoints[:,0], waypoints[:,1], color = "grey", marker = '^', s = 100, alpha=0.5, label='Waypoints')
# cbar.set_label("Sea Surface Temperature [C$^\circ$]")
ax1.set(xlim=[-77.5, -64], ylim=[34, 44], xlabel='Longitude', ylabel='Latitude')
ax1.set_title("2024-06-07 Sea Surface Temperature [C$^\circ$]", size = 15)
ax1.legend(loc = 'upper right')

# ----------------------------------------------------------------------------------------------------------------------------
### Animation
def deploy(point, artist, gps, frame):
    if frame > findframe(gps = gps, point = point):
        x = point[0]
        y = point[1]
        return artist.set_offsets(np.stack([x, y]).T)

def retreave(point, artist, gps, frame):
    if frame > findframe(gps = gps, point = point):
        x = np.nan
        y = np.nan
        return artist.set_offsets(np.stack([x, y]).T)

def update(frame):
    # for each frame, update the data stored on each artist.
    u = u0[(frame-5):frame]
    v = v0[(frame-5):frame]
    # update the scatter plot:
    data = np.stack([u, v]).T
    scat.set_offsets(data)

    ### CTDs.
    #deploy(point =  ctd[0,:], artist = ctd1, gps = pos, frame = frame)
    #deploy(point =  ctd[1,:], artist = ctd2, gps = pos, frame = frame)
    #deploy(point =  ctd[2,:], artist = ctd3, gps = pos, frame = frame)
    #deploy(point =  ctd[3,:], artist = ctd4, gps = pos, frame = frame)
    #deploy(point =  ctd[4,:], artist = ctd5, gps = pos, frame = frame)
    #deploy(point =  ctd[5,:], artist = ctd6, gps = pos, frame = frame)
    #deploy(point =  ctd[6,:], artist = ctd7, gps = pos, frame = frame)
    #deploy(point =  ctd[7,:], artist = ctd8, gps = pos, frame = frame)
    #deploy(point =  ctd[8,:], artist = ctd9, gps = pos, frame = frame)
    #deploy(point =  ctd[9,:], artist = ctd10, gps = pos, frame = frame)
    #deploy(point =  ctd[10,:], artist = ctd11, gps = pos, frame = frame)
    #deploy(point =  ctd[11,:], artist = ctd12, gps = pos, frame = frame)
    #deploy(point =  ctd[12,:], artist = ctd13, gps = pos, frame = frame)
    #deploy(point =  ctd[13,:], artist = ctd14, gps = pos, frame = frame)
    #deploy(point =  ctd[14,:], artist = ctd15, gps = pos, frame = frame)
    #deploy(point =  ctd[15,:], artist = ctd16, gps = pos, frame = frame)
    
    ### Pioneer Pies.
    deploy(point =  pioneer_pos[0,:], artist = pion1, gps = pos, frame = frame)
    deploy(point =  pioneer_pos[1,:], artist = pion2, gps = pos, frame = frame)

    # Retreaved
    ### Swot pies.
    retreave(point =  swot_pos[0,:], artist = swot1, gps = pos, frame = frame)
    retreave(point =  swot_pos[1,:], artist = swot2, gps = pos, frame = frame)
    retreave(point =  swot_pos[2,:], artist = swot3, gps = pos, frame = frame)
    retreave(point =  swot_pos[3,:], artist = swot4, gps = pos, frame = frame)
    retreave(point =  swot_pos[4,:], artist = swot5, gps = pos, frame = frame)
    retreave(point =  swot_pos[5,:], artist = swot6, gps = pos, frame = frame)
# I may need to return something from this function. 

ani = animation.FuncAnimation(fig=fig, func=update, frames=s[0], interval=0.1)
plt.show()

# FFwriter = animation.FFMpegWriter(fps=10)
# ani.save('animation.mp4', writer = FFwriter)
