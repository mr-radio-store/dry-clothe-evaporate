import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ================================
# Clothe-drying simulation with evaporate model
# 1. Settings
# ================================
total_hours = 8           # total drying duration
dt = 1                    # timestep in minutes
nt = total_hours*60       # total timesteps

# Cloth parameters
m0 = 0.5                  # initial water mass (kg)
m_final = 0.02            # almost dry threshold (kg)

# Base environmental conditions
T_base = 30               # °C
RH_base = 40              # %
wind_base = 2             # m/s
solar_base = 600          # W/m²

# Sun angle (simulate morning -> midday)
sun_angle = np.linspace(20, 80, nt)

# Environmental variability
cloud_factor = np.random.uniform(0.5, 1.0, nt)       # cloud cover reduces solar
wind_variation = wind_base + np.random.randn(nt)*0.5  # wind gusts
T_variation = T_base + np.random.randn(nt)*2          # temperature fluctuation
RH_variation = RH_base + np.random.randn(nt)*5        # humidity fluctuation

# ================================
# 2. Simulation loop
# ================================
water_mass = np.zeros(nt)
water_mass[0] = m0

# Save snapshots for video every 5 minutes
snapshots = []

for t in range(1, nt):
    # Effective solar based on sun angle & cloud cover
    solar_effective = solar_base * np.sin(np.radians(sun_angle[t])) * cloud_factor[t]
    
    # Dynamic effective drying rate (first-order exponential decay)
    k_eff = 0.0005*T_variation[t] + 0.0003*(wind_variation[t]*10) \
            + 0.0004*(solar_effective/100) - 0.0002*(RH_variation[t]/10)
    k_eff = max(k_eff, 0)
    
    # Update water mass
    dm = -k_eff * water_mass[t-1] * dt
    water_mass[t] = max(water_mass[t-1] + dm, m_final)
    
    # Save snapshot for video every 5 minutes
    if t % 5 == 0:
        snapshots.append(water_mass[t])

# ================================
# 3. Animation (MP4)
# ================================
fig, ax = plt.subplots(figsize=(8,5))
time_snap = np.arange(len(snapshots))*5 / 60  # in hours
line, = ax.plot([], [], color='blue', lw=2)
ax.set_xlim(0, total_hours)
ax.set_ylim(0, m0*1.05)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Water Mass (kg)")
ax.set_title("Cloth Drying Simulation")
ax.grid(True, linestyle='--', alpha=0.5)

def animate(i):
    line.set_data(time_snap[:i+1], snapshots[:i+1])
    ax.set_title(f"Cloth Drying Simulation: {time_snap[i]:.2f} hours")
    return line,

anim = FuncAnimation(fig, animate, frames=len(snapshots), interval=50)
writer = FFMpegWriter(fps=20)
anim.save("cloth_drying_simulation.mp4", writer=writer)
plt.close()

# ================================
# 4. Plot tracking graph (JPEG)
# ================================
plt.figure(figsize=(8,5))
plt.plot(np.arange(nt)/60, water_mass, color='blue')
plt.xlabel("Time (hours)")
plt.ylabel("Water Mass (kg)")
plt.title("Cloth Drying Progress")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("cloth_drying_progress.jpeg", dpi=150)
plt.show()

# ================================
# 5. Estimate drying time
# ================================
dry_index = np.where(water_mass <= m_final)[0]
if len(dry_index) > 0:
    t_dry = dry_index[0]*dt/60
    print(f"Estimated drying time: {t_dry:.2f} hours")
else:
    print("Cloth did not reach dry threshold within simulation time.")
