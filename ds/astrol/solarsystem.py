
from sunpy.coordinates import get_body_heliographic_stonyhurst
from astropy.time import Time
from astropy.coordinates import get_body
import numpy as np
import math
from datetime import datetime

gplanets = ['mercury', 'venus', 'moon', 'mars',
           'jupiter', 'saturn', 'uranus', 'neptune']

def getPlanetPositions(t, planets=None, heliocentric=False):
    planets = gplanets if planets is None else planets
    planet_locations = {}
    for planet in planets:
        planet_locations[planet] = (get_body if not heliocentric else get_body_heliographic_stonyhurst)(planet, time=t)
    return planet_locations