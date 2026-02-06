import ugradio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rtlsdr import RtlSdr # From ug/radio/ugradio_code/src/sdr.py
import asyncio
import time
sdr = ugradio.sdr.SDR() # Creating a ugradio.sdr.SDR object
data = sdr.capture_date()
np.saves('NAME_HERE', data)