# ———————————————————— Imports ———————————————————————————————————————————————————————————————————————————————————————————————————————————
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal 
from scipy.signal import correlate, correlation_lags


# ———————————————————— LOADING DATA Nyquist Zone 0 ————————————————————————————————————————————————————————————————————————————————
# Loading data at 150 kHz, 250 kHz, 350 kHz
with np.load("Z0_150kHz.npz") as data:
    # print(data.files)
    Z0_150 = data['arr_0']
    # print(Z0_150)  
with np.load("Z0_250kHz.npz") as data:
    # print(data.files)
    Z0_250 = data['arr_0']
    # print(Z0_250)
with np.load("Z0_350kHz.npz") as data:
    # print(data.files)
    Z0_350 = data['arr_0']
    # print(Z0_350)

# ———————————————————— LOADING DATA Nyquist Zone 1 ————————————————————————————————————————————————————————————————————————————————
# Loading data at 650 kHz, 750 kHz, 850 kHz
with np.load("Z1_650kHz.npz") as data:
    # print(data.files)
    Z1_650 = data['arr_0']
    # print(Z1_650)
    
with np.load("Z1_750kHz.npz") as data:
    # print(data.files)
    Z1_750 = data['arr_0']
    # print(Z1_750)

with np.load("Z1_850kHz.npz") as data:
    # print(data.files)
    Z1_850 = data['arr_0']
    # print(Z1_850)

# ———————————————————— LOADING DATA Nyquist Zone 2 ————————————————————————————————————————————————————————————————————————————————
# Loading data at 1150 kHz, 1250 kHz, 1350 kHz
with np.load("Z2_1150kHz.npz") as data:
    # print(data.files)
    Z2_1150 = data['arr_0']
    # print(Z2_1150)   
with np.load("Z2_1250kHz.npz") as data:
    # print(data.files)
    Z2_1250 = data['arr_0']
    # print(Z2_1250)
with np.load("Z2_1350kHz.npz") as data:
    # print(data.files)
    Z2_1350 = data['arr_0']
    # print(Z2_1350)

# ———————————————————— LOADING DATA Nyquist Zone 3 ————————————————————————————————————————————————————————————————————————————————
# Loading data at 1650 kHz, 1750 kHz, 1850 kHz
with np.load("Z3_1650kHz.npz") as data:
    # print(data.files)
    Z3_1650 = data['arr_0']
    # print(Z3_1650)
with np.load("Z3_1750kHz.npz") as data:
    # print(data.files)
    Z3_1750 = data['arr_0']
    # print(Z3_1750)
with np.load("Z3_1850kHz.npz") as data:
    # print(data.files)
    Z3_1850 = data['arr_0']
    # print(Z3_1850)


# ———————————————————— DIGITAL SAMPLING AND NYQUIST CRITERION ————————————————————————————————————————————————————————————————————————————————
# Sample Rate to Time Conversion Function
def N2time(v, N):  # Takes the length of the array for each sample rate and divides it by the sampling rate, giving time
    return np.arange(N) / v

# Sample Rate Conversion for each time
sample_rate = 1.0e6

t_Z0_150 = N2time(sample_rate, len(Z0_150[1]))
t_Z0_250 = N2time(sample_rate, len(Z0_250[1]))
t_Z0_350 = N2time(sample_rate, len(Z0_350[1]))

t_Z1_650 = N2time(sample_rate, len(Z1_650[1]))
t_Z1_750 = N2time(sample_rate, len(Z1_750[1]))
t_Z1_850 = N2time(sample_rate, len(Z1_850[1]))

t_Z2_1150 = N2time(sample_rate, len(Z2_1150[1]))
t_Z2_1250 = N2time(sample_rate, len(Z2_1250[1]))
t_Z2_1350 = N2time(sample_rate, len(Z2_1350[1]))

t_Z3_1650 = N2time(sample_rate, len(Z3_1650[1]))
t_Z3_1750 = N2time(sample_rate, len(Z3_1750[1]))
t_Z3_1850 = N2time(sample_rate, len(Z3_1850[1]))


# ———————————————————— VOLTAGE SPECTRUM ————————————————————————————————————————————————————————————————————————————————————————————————————

dt = 1 / sample_rate # Consistent time since only one sampling rate

# ———————————————————— Nyquist Zone 0 ————————————————————
Xf_150 = np.fft.fft(Z0_150[1]) # Frequency bins
freq150 = np.fft.fftfreq(len(Z0_150[1]), d=dt) # Generates frequency values in each bin in Xf
Xf_250 = np.fft.fft(Z0_250[1])
freq250 = np.fft.fftfreq(len(Z0_250[1]), d=dt)
Xf_350 = np.fft.fft(Z0_350[1])
freq350 = np.fft.fftfreq(len(Z0_350[1]), d=dt)
V_150 = np.abs(Xf_150)
V_250 = np.abs(Xf_250)
V_350 = np.abs(Xf_350)

# ———————————————————— Nyquist Zone 1 ————————————————————
Xf_650 = np.fft.fft(Z1_650[1]) # Frequency bins
freq650 = np.fft.fftfreq(len(Z1_650[1]), d=dt) # Generates frequency values in each bin in Xf
Xf_750 = np.fft.fft(Z1_750[1])
freq750 = np.fft.fftfreq(len(Z1_750[1]), d=dt)
Xf_850 = np.fft.fft(Z1_850[1])
freq850 = np.fft.fftfreq(len(Z1_850[1]), d=dt)
V_650 = np.abs(Xf_650)
V_750 = np.abs(Xf_750)
V_850 = np.abs(Xf_850)

# ———————————————————— Nyquist Zone 2 ————————————————————
Xf_1150 = np.fft.fft(Z2_1150[1]) # Frequency bins
freq1150 = np.fft.fftfreq(len(Z2_1150[1]), d=dt) # Generates frequency values in each bin in Xf
Xf_1250 = np.fft.fft(Z2_1250[1])
freq1250 = np.fft.fftfreq(len(Z2_1250[1]), d=dt)
Xf_1350 = np.fft.fft(Z2_1350[1])
freq1350 = np.fft.fftfreq(len(Z2_1350[1]), d=dt)
V_1150 = np.abs(Xf_1150)
V_1250 = np.abs(Xf_1250)
V_1350 = np.abs(Xf_1350)

# ———————————————————— Nyquist Zone 3 ————————————————————
Xf_1650 = np.fft.fft(Z3_1650[1]) # Frequency bins
freq1650 = np.fft.fftfreq(len(Z3_1650[1]), d=dt)
Xf_1750 = np.fft.fft(Z3_1750[1])
freq1750 = np.fft.fftfreq(len(Z3_1750[1]), d=dt)
Xf_1850 = np.fft.fft(Z3_1850[1])
freq1850 = np.fft.fftfreq(len(Z3_1850[1]), d=dt)
V_1650 = np.abs(Xf_1650)
V_1750 = np.abs(Xf_1750)
V_1850 = np.abs(Xf_1850)


# ———————————————————— COMPLEX VOLTAGE SPECTRUM ————————————————————————————————————————————————————————————————————————————————————————————————————

# ———————————————————— Nyquist Zone 0 ————————————————————
Shift_Xf_150 = np.fft.fftshift(Xf_150) # Frequency bins
Shift_freq150 = np.fft.fftshift(freq150) # Generates frequency values in each bin in Xf
Shift_Xf_250 = np.fft.fftshift(Xf_250)
Shift_freq250 = np.fft.fftshift(freq250)
Shift_Xf_350 = np.fft.fftshift(Xf_350)
Shift_freq350 = np.fft.fftshift(freq350)
Shift_V_150 = np.abs(Shift_Xf_150)
Shift_V_250 = np.abs(Shift_Xf_250)
Shift_V_350 = np.abs(Shift_Xf_350)

# ———————————————————— Nyquist Zone 1 ————————————————————
Shift_Xf_650 = np.fft.fftshift(Xf_650)
Shift_freq650 = np.fft.fftshift(freq650)
Shift_Xf_750 = np.fft.fftshift(Xf_750)
Shift_freq750 = np.fft.fftshift(freq750)
Shift_Xf_850 = np.fft.fftshift(Xf_850)
Shift_freq850 = np.fft.fftshift(freq850)
Shift_V_650 = np.abs(Shift_Xf_650)
Shift_V_750 = np.abs(Shift_Xf_750)
Shift_V_850 = np.abs(Shift_Xf_850)

# ———————————————————— Nyquist Zone 2 ————————————————————
Shift_Xf_1150 = np.fft.fftshift(Xf_1150) # Frequency bins
Shift_freq1150 = np.fft.fftshift(freq1150) # Generates frequency values in each bin in Xf
Shift_Xf_1250 = np.fft.fftshift(Xf_1250)
Shift_freq1250 = np.fft.fftshift(freq1250)
Shift_Xf_1350 = np.fft.fftshift(Xf_1350)
Shift_freq1350 = np.fft.fftshift(freq1350)
Shift_V_1150 = np.abs(Shift_Xf_1150)
Shift_V_1250 = np.abs(Shift_Xf_1250)
Shift_V_1350 = np.abs(Shift_Xf_1350)

# ———————————————————— Nyquist Zone 3 ————————————————————
Shift_Xf_1650 = np.fft.fftshift(Xf_1650)
Shift_freq1650 = np.fft.fftshift(freq1650)
Shift_Xf_1750 = np.fft.fftshift(Xf_1750)
Shift_freq1750 = np.fft.fftshift(freq1750)
Shift_Xf_1850 = np.fft.fftshift(Xf_1850)
Shift_freq1850 = np.fft.fftshift(freq1850)
Shift_V_1650 = np.abs(Shift_Xf_1650)
Shift_V_1750 = np.abs(Shift_Xf_1750)
Shift_V_1850 = np.abs(Shift_Xf_1850)


# ———————————————————— POWER SPECTRUM ————————————————————————————————————————————————————————————————————————————————————————————————————
Shift_P_150  = np.abs(Shift_Xf_150)**2
Shift_P_250  = np.abs(Shift_Xf_250)**2
Shift_P_350  = np.abs(Shift_Xf_350)**2

Shift_P_650  = np.abs(Shift_Xf_650)**2
Shift_P_750  = np.abs(Shift_Xf_750)**2
Shift_P_850  = np.abs(Shift_Xf_850)**2

Shift_P_1150 = np.abs(Shift_Xf_1150)**2
Shift_P_1250 = np.abs(Shift_Xf_1250)**2
Shift_P_1350 = np.abs(Shift_Xf_1350)**2

Shift_P_1650 = np.abs(Shift_Xf_1650)**2
Shift_P_1750 = np.abs(Shift_Xf_1750)**2
Shift_P_1850 = np.abs(Shift_Xf_1850)**2


# ———————————————————— Fourier Transform of a power series ————————————————————————————————————————————————————————————————————————————————
# We want to use the UNSHIFTED data for our inverse fourier transform! 

InverseFT_150 = np.fft.ifft(Xf_150)
InverseFT_250 = np.fft.ifft(Xf_250)
InverseFT_350 = np.fft.ifft(Xf_350)

InverseFT_650 = np.fft.ifft(Xf_650)
InverseFT_750 = np.fft.ifft(Xf_750)
InverseFT_850 = np.fft.ifft(Xf_850)

InverseFT_1150 = np.fft.ifft(Xf_1150)
InverseFT_1250 = np.fft.ifft(Xf_1250)
InverseFT_1350 = np.fft.ifft(Xf_1350)

InverseFT_1650 = np.fft.ifft(Xf_1650)
InverseFT_1750 = np.fft.ifft(Xf_1750)
InverseFT_1850 = np.fft.ifft(Xf_1850)


# ———————————————————— ACF FUNCTION ————————————————————————————————————————————————————————————————————————————————
# Voltage 150
Voltage_150 = Z0_150[1] - np.mean(Z0_150[1])
dt_150 = np.mean(np.diff(t_Z0_150))
sample_rate_150 = 1.0 / dt_150

N_150 = len(Voltage_150)
N_pad_150 = (2 * N_150) - 1

ACF_150 = np.correlate(Voltage_150, Voltage_150, mode='full')
ACF_150 = ACF_150 / ACF_150[len(ACF_150)//2]

V_pad_150 = np.zeros(N_pad_150, dtype=complex)
V_pad_150[:N_150] = Voltage_150

FFT_V_pad_150 = np.fft.fft(V_pad_150)
Power_150 = np.abs(FFT_V_pad_150)**2
ACF_powspec_150 = np.fft.ifft(Power_150).real
ACF_powspec_150 = np.fft.fftshift(ACF_powspec_150)
ACF_powspec_150 = ACF_powspec_150 / ACF_powspec_150[len(ACF_powspec_150)//2]

lags_150 = np.arange(-(N_150-1), N_150) / sample_rate_150

# Voltage 250
Voltage_250 = Z0_250[1] - np.mean(Z0_250[1])
dt_250 = np.mean(np.diff(t_Z0_250))
sample_rate_250 = 1.0 / dt_250

N_250 = len(Voltage_250)
N_pad_250 = (2 * N_250) - 1

ACF_250 = np.correlate(Voltage_250, Voltage_250, mode='full')
ACF_250 = ACF_250 / ACF_250[len(ACF_250)//2]

V_pad_250 = np.zeros(N_pad_250, dtype=complex)
V_pad_250[:N_250] = Voltage_250

FFT_V_pad_250 = np.fft.fft(V_pad_250)
Power_250 = np.abs(FFT_V_pad_250)**2
ACF_powspec_250 = np.fft.ifft(Power_250).real
ACF_powspec_250 = np.fft.fftshift(ACF_powspec_250)
ACF_powspec_250 = ACF_powspec_250 / ACF_powspec_250[len(ACF_powspec_250)//2]

lags_250 = np.arange(-(N_250-1), N_250) / sample_rate_250

# Voltage 350
Voltage_350 = Z0_350[1] - np.mean(Z0_350[1])
dt_350 = np.mean(np.diff(t_Z0_350))
sample_rate_350 = 1.0 / dt_350

N_350 = len(Voltage_350)
N_pad_350 = (2 * N_350) - 1

ACF_350 = np.correlate(Voltage_350, Voltage_350, mode='full')
ACF_350 = ACF_350 / ACF_350[len(ACF_350)//2]

V_pad_350 = np.zeros(N_pad_350, dtype=complex)
V_pad_350[:N_350] = Voltage_350

FFT_V_pad_350 = np.fft.fft(V_pad_350)
Power_350 = np.abs(FFT_V_pad_350)**2
ACF_powspec_350 = np.fft.ifft(Power_350).real
ACF_powspec_350 = np.fft.fftshift(ACF_powspec_350)
ACF_powspec_350 = ACF_powspec_350 / ACF_powspec_350[len(ACF_powspec_350)//2]

lags_350 = np.arange(-(N_350-1), N_350) / sample_rate_350

# Voltage 650
Voltage_650 = Z1_650[1] - np.mean(Z1_650[1])
dt_650 = np.mean(np.diff(t_Z1_650))
sample_rate_650 = 1.0 / dt_650

N_650 = len(Voltage_650)
N_pad_650 = (2 * N_650) - 1

ACF_650 = np.correlate(Voltage_650, Voltage_650, mode='full')
ACF_650 = ACF_650 / ACF_650[len(ACF_650)//2]

V_pad_650 = np.zeros(N_pad_650, dtype=complex)
V_pad_650[:N_650] = Voltage_650

FFT_V_pad_650 = np.fft.fft(V_pad_650)
Power_650 = np.abs(FFT_V_pad_650)**2
ACF_powspec_650 = np.fft.ifft(Power_650).real
ACF_powspec_650 = np.fft.fftshift(ACF_powspec_650)
ACF_powspec_650 = ACF_powspec_650 / ACF_powspec_650[len(ACF_powspec_650)//2]

lags_650 = np.arange(-(N_650-1), N_650) / sample_rate_650

# Voltage 750
Voltage_750 = Z1_750[1] - np.mean(Z1_750[1])
dt_750 = np.mean(np.diff(t_Z1_750))
sample_rate_750 = 1.0 / dt_750

N_750 = len(Voltage_750)
N_pad_750 = (2 * N_750) - 1

ACF_750 = np.correlate(Voltage_750, Voltage_750, mode='full')
ACF_750 = ACF_750 / ACF_750[len(ACF_750)//2]

V_pad_750 = np.zeros(N_pad_750, dtype=complex)
V_pad_750[:N_750] = Voltage_750

FFT_V_pad_750 = np.fft.fft(V_pad_750)
Power_750 = np.abs(FFT_V_pad_750)**2
ACF_powspec_750 = np.fft.ifft(Power_750).real
ACF_powspec_750 = np.fft.fftshift(ACF_powspec_750)
ACF_powspec_750 = ACF_powspec_750 / ACF_powspec_750[len(ACF_powspec_750)//2]

lags_750 = np.arange(-(N_750-1), N_750) / sample_rate_750

# Voltage 850
Voltage_850 = Z1_850[1] - np.mean(Z1_850[1])
dt_850 = np.mean(np.diff(t_Z1_850))
sample_rate_850 = 1.0 / dt_850

N_850 = len(Voltage_850)
N_pad_850 = (2 * N_850) - 1

ACF_850 = np.correlate(Voltage_850, Voltage_850, mode='full')
ACF_850 = ACF_850 / ACF_850[len(ACF_850)//2]

V_pad_850 = np.zeros(N_pad_850, dtype=complex)
V_pad_850[:N_850] = Voltage_850

FFT_V_pad_850 = np.fft.fft(V_pad_850)
Power_850 = np.abs(FFT_V_pad_850)**2
ACF_powspec_850 = np.fft.ifft(Power_850).real
ACF_powspec_850 = np.fft.fftshift(ACF_powspec_850)
ACF_powspec_850 = ACF_powspec_850 / ACF_powspec_850[len(ACF_powspec_850)//2]

lags_850 = np.arange(-(N_850-1), N_850) / sample_rate_850

# Voltage 1150
Voltage_1150 = Z2_1150[1] - np.mean(Z2_1150[1])
dt_1150 = np.mean(np.diff(t_Z2_1150))
sample_rate_1150 = 1.0 / dt_1150

N_1150 = len(Voltage_1150)
N_pad_1150 = (2 * N_1150) - 1

ACF_1150 = np.correlate(Voltage_1150, Voltage_1150, mode='full')
ACF_1150 = ACF_1150 / ACF_1150[len(ACF_1150)//2]

V_pad_1150 = np.zeros(N_pad_1150, dtype=complex)
V_pad_1150[:N_1150] = Voltage_1150

FFT_V_pad_1150 = np.fft.fft(V_pad_1150)
Power_1150 = np.abs(FFT_V_pad_1150)**2
ACF_powspec_1150 = np.fft.ifft(Power_1150).real
ACF_powspec_1150 = np.fft.fftshift(ACF_powspec_1150)
ACF_powspec_1150 = ACF_powspec_1150 / ACF_powspec_1150[len(ACF_powspec_1150)//2]

lags_1150 = np.arange(-(N_1150-1), N_1150) / sample_rate_1150

# Voltage 1250
Voltage_1250 = Z2_1250[1] - np.mean(Z2_1250[1])
dt_1250 = np.mean(np.diff(t_Z2_1250))
sample_rate_1250 = 1.0 / dt_1250

N_1250 = len(Voltage_1250)
N_pad_1250 = (2 * N_1250) - 1

ACF_1250 = np.correlate(Voltage_1250, Voltage_1250, mode='full')
ACF_1250 = ACF_1250 / ACF_1250[len(ACF_1250)//2]

V_pad_1250 = np.zeros(N_pad_1250, dtype=complex)
V_pad_1250[:N_1250] = Voltage_1250

FFT_V_pad_1250 = np.fft.fft(V_pad_1250)
Power_1250 = np.abs(FFT_V_pad_1250)**2
ACF_powspec_1250 = np.fft.ifft(Power_1250).real
ACF_powspec_1250 = np.fft.fftshift(ACF_powspec_1250)
ACF_powspec_1250 = ACF_powspec_1250 / ACF_powspec_1250[len(ACF_powspec_1250)//2]

lags_1250 = np.arange(-(N_1250-1), N_1250) / sample_rate_1250

# Voltage 1350
Voltage_1350 = Z2_1350[1] - np.mean(Z2_1350[1])
dt_1350 = np.mean(np.diff(t_Z2_1350))
sample_rate_1350 = 1.0 / dt_1350

N_1350 = len(Voltage_1350)
N_pad_1350 = (2 * N_1350) - 1

ACF_1350 = np.correlate(Voltage_1350, Voltage_1350, mode='full')
ACF_1350 = ACF_1350 / ACF_1350[len(ACF_1350)//2]

V_pad_1350 = np.zeros(N_pad_1350, dtype=complex)
V_pad_1350[:N_1350] = Voltage_1350

FFT_V_pad_1350 = np.fft.fft(V_pad_1350)
Power_1350 = np.abs(FFT_V_pad_1350)**2
ACF_powspec_1350 = np.fft.ifft(Power_1350).real
ACF_powspec_1350 = np.fft.fftshift(ACF_powspec_1350)
ACF_powspec_1350 = ACF_powspec_1350 / ACF_powspec_1350[len(ACF_powspec_1350)//2]

lags_1350 = np.arange(-(N_1350-1), N_1350) / sample_rate_1350

# Voltage 1650
Voltage_1650 = Z3_1650[1] - np.mean(Z3_1650[1])
dt_1650 = np.mean(np.diff(t_Z3_1650))
sample_rate_1650 = 1.0 / dt_1650

N_1650 = len(Voltage_1650)
N_pad_1650 = (2 * N_1650) - 1

ACF_1650 = np.correlate(Voltage_1650, Voltage_1650, mode='full')
ACF_1650 = ACF_1650 / ACF_1650[len(ACF_1650)//2]

V_pad_1650 = np.zeros(N_pad_1650, dtype=complex)
V_pad_1650[:N_1650] = Voltage_1650

FFT_V_pad_1650 = np.fft.fft(V_pad_1650)
Power_1650 = np.abs(FFT_V_pad_1650)**2
ACF_powspec_1650 = np.fft.ifft(Power_1650).real
ACF_powspec_1650 = np.fft.fftshift(ACF_powspec_1650)
ACF_powspec_1650 = ACF_powspec_1650 / ACF_powspec_1650[len(ACF_powspec_1650)//2]

lags_1650 = np.arange(-(N_1650-1), N_1650) / sample_rate_1650

# Voltage 1750
Voltage_1750 = Z3_1750[1] - np.mean(Z3_1750[1])
dt_1750 = np.mean(np.diff(t_Z3_1750))
sample_rate_1750 = 1.0 / dt_1750

N_1750 = len(Voltage_1750)
N_pad_1750 = (2 * N_1750) - 1

ACF_1750 = np.correlate(Voltage_1750, Voltage_1750, mode='full')
ACF_1750 = ACF_1750 / ACF_1750[len(ACF_1750)//2]

V_pad_1750 = np.zeros(N_pad_1750, dtype=complex)
V_pad_1750[:N_1750] = Voltage_1750

FFT_V_pad_1750 = np.fft.fft(V_pad_1750)
Power_1750 = np.abs(FFT_V_pad_1750)**2
ACF_powspec_1750 = np.fft.ifft(Power_1750).real
ACF_powspec_1750 = np.fft.fftshift(ACF_powspec_1750)
ACF_powspec_1750 = ACF_powspec_1750 / ACF_powspec_1750[len(ACF_powspec_1750)//2]

lags_1750 = np.arange(-(N_1750-1), N_1750) / sample_rate_1750

# Voltage 1850
Voltage_1850 = Z3_1850[1] - np.mean(Z3_1850[1])
dt_1850 = np.mean(np.diff(t_Z3_1850))
sample_rate_1850 = 1.0 / dt_1850

N_1850 = len(Voltage_1850)
N_pad_1850 = (2 * N_1850) - 1

ACF_1850 = np.correlate(Voltage_1850, Voltage_1850, mode='full')
ACF_1850 = ACF_1850 / ACF_1850[len(ACF_1850)//2]

V_pad_1850 = np.zeros(N_pad_1850, dtype=complex)
V_pad_1850[:N_1850] = Voltage_1850

FFT_V_pad_1850 = np.fft.fft(V_pad_1850)
Power_1850 = np.abs(FFT_V_pad_1850)**2
ACF_powspec_1850 = np.fft.ifft(Power_1850).real
ACF_powspec_1850 = np.fft.fftshift(ACF_powspec_1850)
ACF_powspec_1850 = ACF_powspec_1850 / ACF_powspec_1850[len(ACF_powspec_1850)//2]

lags_1850 = np.arange(-(N_1850-1), N_1850) / sample_rate_1850


# ———————————————————— Leakage Power ————————————————————————————————————————————————————————————————————————————————


# ———————————————————— LOADING DATA Nyquist Zone 0 ————————————————————————————————————————————————————————————————————————————————
# Loading data for Double-SideBand (DSB) Mixer
# with np.load("7.1.1.npz") as data:
#     dsb_1500kHz = data1['data_direct']
#     signal = data_direct1[trial_idx, channel_idx,;
#     # print(Z0_150)

# with np.load("7.1.2.npz") as data:
#     # print(data.files)
#     dsb_1575kHz = data['arr_0']
#     # print(Z0_150)

# channel_idx = 0
# trial_idx = 0