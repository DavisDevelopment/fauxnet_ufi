import numpy as np
import math

from numpy import *
from datetime import datetime as dt

def position_from_sun(d=None):
   d = d or dt.now()
   day_of_year = d.timetuple().tm_yday
   
   return day_of_year