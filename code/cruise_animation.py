import matplotlib.pyplot as plt
from erddapClient import ERDDAP_Griddap
from netCDF4 import Dataset
from matplotlib import colormaps
from pathlib import Path
import numpy as np
from pycurrents.file.binfile_n import BinfileSet
import xarray as xr
import matplotlib.animation as animation
import pandas as pd

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

def findframe(gps, point, eyballit=0.01):
    A = gps[:,0] > point[0] - eyballit
    B = gps[:,0] < point[0] + eyballit
    C = gps[:,1] > point[1] - eyballit
    D = gps[:,1] < point[1] + eyballit
    a = np.logical_and(A , B)
    b = np.logical_and(C , D)
    c = np.logical_and(a , b)
    idx = np.where(c)
    try:
        frame = np.min(idx)
        return(frame)
    except:
        return 9999

# ----------------------------------------------------------------------------------------------------------------------------
# Sea-Surface Temperature, NOAA Geo-polar Blended Analysis Day+Night, GHRSST, Near Real-Time, Global 5km, 2019-Present, Daily 
file_id = Dataset('/home/jamie/projects/rogerrevelle/data/sattalite/noaacwBLENDEDsstDNDaily_8c1a_9bf0_afc5_U1718024700052.nc') 
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

waypoints = np.column_stack(([-71.0, -70.67], [39.5, 41.527]))
waypoints = np.array(waypoints)

# ----------------------------------------------------------------------------------------------------------------------------
### CTD and XBT locations. 

#ctd = np.genfromtxt('../data/CTD_positions.csv', delimiter=',')
#xbt = np.genfromtxt('../data/XBT_positions.csv', delimiter=',')

xbt = pd.read_csv('../data/eventlog/XBT.csv')
ctd = pd.read_csv('../data/eventlog/CTD.csv')
pies = pd.read_csv('../data/eventlog/CPIES.csv')
dws_drift = pd.read_csv('../data/eventlog/dws_drifter.csv')
svt_drift = pd.read_csv('../data/eventlog/svt_drifter.csv')

ctd = np.array(ctd[["Longitude", "Latitude"]])
xbt = np.array(xbt[["Longitude", "Latitude"]])
pies = np.array(pies[["Longitude", "Latitude"]])
dws_drift = np.array(dws_drift[["Longitude", "Latitude"]])
svt_drift = np.array(svt_drift[["Longitude", "Latitude"]])

# ----------------------------------------------------------------------------------------------------------------------------
### SHIP GPS Positions
sea = readrbins(pth ='../data/rbins/', sensor = 'seapath380', tag = 'gps')
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
scat = ax1.scatter(u0[0], v0[0], s = 100, alpha = blur, color = "black")

# Initillize the drifter plot.
drift1 = ax1.scatter(tmp[0,0], tmp[0,1], color = "black",  marker = 'o', s = 25, alpha = 0.7, label='DWS Drifters')
drift2 = ax1.scatter(tmp[0,0], tmp[0,1], color = "black",  marker = 'o', s = 25, alpha = 0.7)
drift3 = ax1.scatter(tmp[0,0], tmp[0,1], color = "black",  marker = 'o', s = 25, alpha = 0.7)
drift4 = ax1.scatter(tmp[0,0], tmp[0,1], color = "black",  marker = 'o', s = 25, alpha = 0.7)
drift5 = ax1.scatter(tmp[0,0], tmp[0,1], color = "black",  marker = 'o', s = 25, alpha = 0.7)
drift6 = ax1.scatter(tmp[0,0], tmp[0,1], color = "black",  marker = 'o', s = 25, alpha = 0.7)
drift7 = ax1.scatter(tmp[0,0], tmp[0,1], color = "black",  marker = 'o', s = 25, alpha = 0.7)

sdrift1 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = 'o', s = 25, alpha = 0.7, label='SVT Drifters')
sdrift2 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = 'o', s = 25, alpha = 0.7)
sdrift3 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = 'o', s = 25, alpha = 0.7)

# Initillize the CTD plot.
ctd1 = ax1.scatter(tmp[0,0], tmp[0,1], color = "black",  marker = '+', s = 50, label='CTD Cast')
ctd2 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd3 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd4 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd5 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd6 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd7 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd8 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd9 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd10 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd11 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd12 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd13 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd14 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")#
ctd15 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd16 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd17 = ax1.scatter(tmp[0,0], tmp[0,1], color = "black",  marker = '+', s = 50)
ctd18 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd19 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd20 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd21 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd22 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd23 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd24 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd25 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd26 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd27 = ax1.scatter(tmp[0,0], tmp[0,1], color = "black",  marker = '+', s = 50)
ctd28 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd29 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd30 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd31 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd32 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd33 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd34 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd35 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd36 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd37 = ax1.scatter(tmp[0,0], tmp[0,1], color = "black",  marker = '+', s = 50)
ctd38 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd39 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd40 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd41 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd42 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd43 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd44 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd45 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd46 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd47 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd48 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd48 = ax1.scatter(tmp[0,0], tmp[0,1], color = "black",  marker = '+', s = 50)
ctd49 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd50 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd51 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd52 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd53 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd54 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd55 = ax1.scatter(tmp[0,0], tmp[0,1], color = "black",  marker = '+', s = 50)
ctd56 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd57 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd58 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd59 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")
ctd60 = ax1.scatter(tmp[0,0], tmp[0,1], s = 50, marker = '+', color = "black")

# Initillize the XBT plot.
xbt1 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25, label='XBT drop')
xbt2 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt3 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt4 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt5 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt6 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt7 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt8 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt9 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt10 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt11 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt12 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt13 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt14 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt15 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt16 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)
xbt17 = ax1.scatter(tmp[0,0], tmp[0,1], color = "grey",  marker = '+', s = 25)

# Deply instruments
pion1 = ax1.scatter(tmp[0,0], tmp[0,1], alpha = 0.7, color = "grey", marker = 'X', s = 100, label='Pioneer PIES')
pion2 = ax1.scatter(tmp[0,0], tmp[0,1], alpha = 0.7, color = "grey", marker = 'X', s = 100)

# Initilize the swot. 
swot1 = ax1.scatter(swot_pos[0,0], swot_pos[0,1], alpha = 0.7, color = "black", marker = 'X', s = 100, label='SWOT PIES')
swot2 = ax1.scatter(swot_pos[1,0], swot_pos[1,1], alpha = 0.7, color = "black", marker = 'X', s = 100)
swot3 = ax1.scatter(swot_pos[2,0], swot_pos[2,1], alpha = 0.7, color = "black", marker = 'X', s = 100)
swot4 = ax1.scatter(swot_pos[3,0], swot_pos[3,1], alpha = 0.7, color = "black", marker = 'X', s = 100)
swot5 = ax1.scatter(swot_pos[4,0], swot_pos[4,1], alpha = 0.7, color = "black", marker = 'X', s = 100)
swot6 = ax1.scatter(swot_pos[5,0], swot_pos[5,1], alpha = 0.7, color = "black", marker = 'X', s = 100)

# CTD points
#ax1.scatter(ctd[:,0], ctd[:,1], color = "black", marker = 'o', s = 100, label='CTD')
ax1.scatter(waypoints[:,0], waypoints[:,1], color = "red", marker = '^', s = 100, alpha=0.7, label='Waypoints')
# cbar.set_label("Sea Surface Temperature [C$^\circ$]")
ax1.set(xlim=[-77.5, -64], ylim=[34, 44], xlabel='Longitude', ylabel='Latitude')
ax1.set_title("2024-06-08 Sea Surface Temperature [C$^\circ$]", size = 15)
ax1.text(-67.5, 34.2, "Jamie Ash UHDAS Currents Group")
ax1.legend(loc = 'upper right')

# ----------------------------------------------------------------------------------------------------------------------------
### Animation
def deploy(point, artist, gps, frame, eyballit=0.01):
    if frame > findframe(gps = gps, point = point, eyballit=eyballit):
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

    ### DRIFTERS
    deploy(point =  dws_drift[0,:], artist = drift1, gps = pos, frame = frame)
    deploy(point =  dws_drift[1,:], artist = drift2, gps = pos, frame = frame)
    deploy(point =  dws_drift[2,:], artist = drift3, gps = pos, frame = frame)
    deploy(point =  dws_drift[3,:], artist = drift4, gps = pos, frame = frame)
    deploy(point =  dws_drift[4,:], artist = drift5, gps = pos, frame = frame)
    deploy(point =  dws_drift[5,:], artist = drift6, gps = pos, frame = frame)
    deploy(point =  dws_drift[6,:], artist = drift7, gps = pos, frame = frame)

    deploy(point =  svt_drift[0,:], artist = sdrift1, gps = pos, frame = frame)
    deploy(point =  svt_drift[1,:], artist = sdrift2, gps = pos, frame = frame)
    deploy(point =  svt_drift[2,:], artist = sdrift3, gps = pos, frame = frame)

    ### CTDs.
    deploy(point =  ctd[0,:], artist = ctd1, gps = pos, frame = frame)
    deploy(point =  ctd[1,:], artist = ctd2, gps = pos, frame = frame)
    deploy(point =  ctd[2,:], artist = ctd3, gps = pos, frame = frame)
    deploy(point =  ctd[3,:], artist = ctd4, gps = pos, frame = frame)
    deploy(point =  ctd[4,:], artist = ctd5, gps = pos, frame = frame)
    deploy(point =  ctd[5,:], artist = ctd6, gps = pos, frame = frame)
    deploy(point =  ctd[6,:], artist = ctd7, gps = pos, frame = frame)
    deploy(point =  ctd[7,:], artist = ctd8, gps = pos, frame = frame)
    deploy(point =  ctd[8,:], artist = ctd9, gps = pos, frame = frame)
    deploy(point =  ctd[9,:], artist = ctd10, gps = pos, frame = frame)
    deploy(point =  ctd[10,:], artist = ctd11, gps = pos, frame = frame)
    deploy(point =  ctd[11,:], artist = ctd12, gps = pos, frame = frame)
    deploy(point =  ctd[12,:], artist = ctd13, gps = pos, frame = frame)
    deploy(point =  ctd[13,:], artist = ctd14, gps = pos, frame = frame)
    deploy(point =  ctd[14,:], artist = ctd15, gps = pos, frame = frame)
    deploy(point =  ctd[15,:], artist = ctd16, gps = pos, frame = frame)
    deploy(point =  ctd[16,:], artist = ctd17, gps = pos, frame = frame)
    deploy(point =  ctd[17,:], artist = ctd18, gps = pos, frame = frame)
    deploy(point =  ctd[18,:], artist = ctd19, gps = pos, frame = frame)
    deploy(point =  ctd[19,:], artist = ctd20, gps = pos, frame = frame)
    deploy(point =  ctd[20,:], artist = ctd21, gps = pos, frame = frame)
    deploy(point =  ctd[21,:], artist = ctd22, gps = pos, frame = frame)
    deploy(point =  ctd[22,:], artist = ctd23, gps = pos, frame = frame)
    deploy(point =  ctd[23,:], artist = ctd24, gps = pos, frame = frame)
    deploy(point =  ctd[24,:], artist = ctd25, gps = pos, frame = frame)
    deploy(point =  ctd[25,:], artist = ctd26, gps = pos, frame = frame)
    deploy(point =  ctd[26,:], artist = ctd27, gps = pos, frame = frame)
    deploy(point =  ctd[27,:], artist = ctd28, gps = pos, frame = frame)
    deploy(point =  ctd[28,:], artist = ctd29, gps = pos, frame = frame)
    deploy(point =  ctd[29,:], artist = ctd30, gps = pos, frame = frame)
    deploy(point =  ctd[30,:], artist = ctd31, gps = pos, frame = frame)
    deploy(point =  ctd[31,:], artist = ctd32, gps = pos, frame = frame)
    deploy(point =  ctd[32,:], artist = ctd33, gps = pos, frame = frame)
    deploy(point =  ctd[33,:], artist = ctd34, gps = pos, frame = frame)
    deploy(point =  ctd[34,:], artist = ctd35, gps = pos, frame = frame)
    deploy(point =  ctd[35,:], artist = ctd36, gps = pos, frame = frame)
    deploy(point =  ctd[36,:], artist = ctd37, gps = pos, frame = frame)
    deploy(point =  ctd[37,:], artist = ctd38, gps = pos, frame = frame)
    deploy(point =  ctd[38,:], artist = ctd39, gps = pos, frame = frame)
    deploy(point =  ctd[39,:], artist = ctd40, gps = pos, frame = frame)
    deploy(point =  ctd[40,:], artist = ctd41, gps = pos, frame = frame)
    deploy(point =  ctd[41,:], artist = ctd42, gps = pos, frame = frame)
    deploy(point =  ctd[42,:], artist = ctd43, gps = pos, frame = frame)
    deploy(point =  ctd[43,:], artist = ctd44, gps = pos, frame = frame)
    deploy(point =  ctd[44,:], artist = ctd45, gps = pos, frame = frame)
    deploy(point =  ctd[45,:], artist = ctd46, gps = pos, frame = frame)
    deploy(point =  ctd[46,:], artist = ctd47, gps = pos, frame = frame)
    deploy(point =  ctd[47,:], artist = ctd48, gps = pos, frame = frame)
    deploy(point =  ctd[48,:], artist = ctd49, gps = pos, frame = frame)
    deploy(point =  ctd[49,:], artist = ctd50, gps = pos, frame = frame)
    deploy(point =  ctd[50,:], artist = ctd51, gps = pos, frame = frame)
    deploy(point =  ctd[51,:], artist = ctd52, gps = pos, frame = frame)
    deploy(point =  ctd[52,:], artist = ctd53, gps = pos, frame = frame)
    deploy(point =  ctd[53,:], artist = ctd54, gps = pos, frame = frame)
    deploy(point =  ctd[54,:], artist = ctd55, gps = pos, frame = frame)
    deploy(point =  ctd[55,:], artist = ctd56, gps = pos, frame = frame)
    deploy(point =  ctd[56,:], artist = ctd57, gps = pos, frame = frame)
    deploy(point =  ctd[57,:], artist = ctd58, gps = pos, frame = frame)
    deploy(point =  ctd[58,:], artist = ctd59, gps = pos, frame = frame)
    deploy(point =  ctd[59,:], artist = ctd60, gps = pos, frame = frame)

    ### XBTs.
    deploy(point =  xbt[0,:], artist = xbt1, gps = pos, frame = frame)
    deploy(point =  xbt[1,:], artist = xbt2, gps = pos, frame = frame)
    deploy(point =  xbt[2,:], artist = xbt3, gps = pos, frame = frame)
    deploy(point =  xbt[3,:], artist = xbt4, gps = pos, frame = frame)
    deploy(point =  xbt[4,:], artist = xbt5, gps = pos, frame = frame)
    deploy(point =  xbt[5,:], artist = xbt6, gps = pos, frame = frame)
    deploy(point =  xbt[6,:], artist = xbt7, gps = pos, frame = frame)
    deploy(point =  xbt[7,:], artist = xbt8, gps = pos, frame = frame)
    deploy(point =  xbt[8,:], artist = xbt9, gps = pos, frame = frame)
    deploy(point =  xbt[9,:], artist = xbt10, gps = pos, frame = frame)
    deploy(point =  xbt[10,:], artist = xbt11, gps = pos, frame = frame)
    deploy(point =  xbt[11,:], artist = xbt12, gps = pos, frame = frame)
    deploy(point =  xbt[12,:], artist = xbt13, gps = pos, frame = frame)
    deploy(point =  xbt[13,:], artist = xbt14, gps = pos, frame = frame)
    deploy(point =  xbt[14,:], artist = xbt15, gps = pos, frame = frame)
    deploy(point =  xbt[15,:], artist = xbt16, gps = pos, frame = frame)
    deploy(point =  xbt[16,:], artist = xbt17, gps = pos, frame = frame)
    
    ### Pioneer Pies.
    deploy(point =  pioneer_pos[0,:], artist = pion1, gps = pos, frame = frame)
    deploy(point =  pioneer_pos[1,:], artist = pion2, gps = pos, frame = frame)

    # Retreaved
    ### Swot pies.
    retreave(point =  swot_pos[0,:], artist = swot1, gps = pos, frame = frame)
    retreave(point =  swot_pos[1,:], artist = swot2, gps = pos, frame = frame)
    #retreave(point =  swot_pos[2,:], artist = swot3, gps = pos, frame = frame)
    retreave(point =  swot_pos[3,:], artist = swot4, gps = pos, frame = frame)
    retreave(point =  swot_pos[4,:], artist = swot5, gps = pos, frame = frame)
    # retreave(point =  swot_pos[5,:], artist = swot6, gps = pos, frame = frame)
# I may need to return something from this function. 

ani = animation.FuncAnimation(fig=fig, func=update, frames=s[0], interval=0.1)
#plt.show()

FFwriter = animation.FFMpegWriter(fps=10)
ani.save('../figures/RR247_movie.mp4', writer = FFwriter)
