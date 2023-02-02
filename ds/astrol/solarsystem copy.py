from datetime import datetime
#!/usr/bin/python3

# Calculate the Julian day at this moment

import sys  # for system connections
import re       # for regex
from time import gmtime, strftime  # for utc
# from . import julian
import julian
# import pandas
# pandas.Timestamp.to_julian_date
import numba
from numba import jit, float32, float64
from numba.experimental import jitclass
from math import *
from collections import namedtuple

def to_jd(D:datetime):
  return julian.to_jd(D)

"""
now let's define some fast-as-fuck helpful functions!
"""


@jit
def fixangle(a): return a - 360.0 * floor(a/360.0)


@jit
def torad(d): return d * pi / 180.0
@jit
def todeg(r): return r * 180.0 / pi
@jit
def dsin(d): return sin(torad(d))
@jit
def dcos(d): return cos(torad(d))


@jit
def kepler(m, ecc):

  epsilon = 1e-6

  m = torad(m)
  e = m
  while 1:
      delta = e - ecc * sin(e) - m
      e = e - delta / (1.0 - ecc * cos(e))

      if abs(delta) <= epsilon:
          break

  return e

class AstronomicalConstants:

  # JDN stands for Julian Day Number
  # Angles here are in degrees

  # 1980 January 0.0 in JDN
  # XXX: DateTime(1980).jdn yields 2444239.5 -- which one is right?
  epoch = 2444238.5

  # Ecliptic longitude of the Sun at epoch 1980.0
  ecliptic_longitude_epoch = 278.833540

  # Ecliptic longitude of the Sun at perigee
  ecliptic_longitude_perigee = 282.596403

  # Eccentricity of Earth's orbit
  earth_eccentricity = 0.016718

  # Semi-major axis of Earth's orbit, in kilometers
  earth_sun_smaxis = 1.49585e8

  # Sun's angular size, in degrees, at semi-major axis distance
  sun_angular_size_smaxis = 0.533128

  ## Elements of the Moon's orbit, epoch 1980.0

  # Moon's mean longitude at the epoch
  luna_mean_longitude_epoch = 64.975464
  # Mean longitude of the perigee at the epoch
  luna_mean_perigee_epoch = 349.383063

  # Mean longitude of the node at the epoch
  node_mean_longitude_epoch = 151.950429

  # Inclination of the Moon's orbit
  luna_inclination = 5.145396

  # Eccentricity of the Moon's orbit
  luna_eccentricity = 0.054900

  # Moon's angular size at distance a from Earth
  luna_angular_size = 0.5181

  # Semi-mojor axis of the Moon's orbit, in kilometers
  luna_smaxis = 384401.0
  
  # Parallax at a distance a from Earth
  luna_parallax = 0.9507

  # Synodic month (new Moon to new Moon), in days
  synodic_month = 29.53058868

  # Base date for E. W. Brown's numbered series of lunations (1923 January 16)
  lunations_base = 2423436.0

  ## Properties of the Earth
  earth_radius = 6378.16

class OrbitalElements:
  # The primary orbital elements are here denoted as:
  N = None # longitude of the ascending node
  i = None # inclination to the ecliptic(plane of the Earth's orbit)
  w = None # argument of perihelion
  a = None # semi-major axis, or mean distance from Sun
  e = None # eccentricity (0=circle, 0-1=ellipse, 1=parabola)
  M = None # mean anomaly (0 at perihelion increases uniformly with time)


  """Related orbital elements are:
  w1 = N + w = longitude of perihelion
  L = M + w1 = mean longitude
  q = a*(1-e) = perihelion distance
  Q = a*(1+e) = aphelion distance
  P = a ^ 1.5 = orbital period(years if a is in AU, astronomical units)
  T = Epoch_of_M - (M(deg)/360_deg) / P = time of perihelion
  v = true anomaly(angle between position and perihelion)
  E = eccentric anomaly
  """
  def __init__(self, N=None, i=None, w=None, a=None, e=None, M=None, d=None):
    N = N
    i = i
    w = w
    a = a
    e = e
    M = M
    E = None
    
    w1 = N + w
    L = M + w1
    q = a * (1 - e)
    Q = a * (1 + e)
    P = a ** 1.5
    ecl = 23.4393 - 3.563E-7 * d
    
    self.__dict__.update(locals())
    #T = ???
      
  def compute_position(self):
    E, a, e, w = self.E, self.a, self.e, self.w
    # To describe the position in the orbit, we use three angles: 
    # Mean Anomaly, True Anomaly, and Eccentric Anomaly. 
    # They are all zero when the planet is in perihelion
    xv = a * ( cos(E) - e ) # = r * cos(v) 
    yv = a * ( sqrt(1.0 - e*e) * sin(E) ) # = r * sin(v) 

    v = atan2( yv, xv )
    r = sqrt( xv*xv + yv*yv )
    
    return dict(v=v, r=r, xv=xv, yv=yv)
    
class Sun(OrbitalElements):
  def __init__(self, **kw):
    OrbitalElements.__init__(self, **kw)
    self.E = self.M + self.e * sin(self.M) * (1.0 + self.e * cos(self.M))
    
  def compute_position(self):
    v, r, xv, yv = tuple(super().compute_position().values())
    ecl = self.ecl
    # Now, compute the Sun's true longitude:
    lonsun = v + self.w

    #Convert lonsun, r to ecliptic rectangular geocentric coordinates xs, ys:
    xs = r * cos(lonsun)
    ys = r * sin(lonsun)
    #(since the Sun always is in the ecliptic plane, zs is of course zero). xs, ys is the Sun's position in a coordinate system in the plane of the ecliptic. To convert this to equatorial, rectangular, geocentric coordinates, compute:
    xe = xs
    ye = ys * cos(ecl)
    ze = ys * sin(ecl)
    #Finally, compute the Sun's Right Ascension(RA) and Declination(Dec):
    RA = atan2(ye, xe)
    Dec = atan2(ze, sqrt(xe*xe+ye*ye))
    
    return dict(lonsun=lonsun, xs=xs, ys=ys, xe=xs, ye=ye, ze=ze, RA=RA, Dec=Dec)
    

# @jitclass
class AstrologicalSolarSystem:
  def __init__(self, _d: float):
    d = self.d = _d
    self.Sun = Sun(N=0.0,
                          i=0.0,
                          w=282.9404 + 4.70935E-5 * d,
                          a=1.000000, # (AU)
                          e=0.016709 - 1.151E-9 * d,
                          M=356.0470 + 0.9856002585 * d,
                          d=d)
    
    self.Moon = OrbitalElements(N=125.1228 - 0.0529538083 * d,
                          i=5.1454,
                          w=318.0634 + 0.1643573223 * d,
                          a=60.2666, # (Earth radii)
                          e=0.054900,
                          M=115.3654 + 13.0649929509 * d,
                          d=d)
    self.Mercury = OrbitalElements(N=48.3313 + 3.24587E-5 * d,
                              i=7.0047 + 5.00E-8 * d,
                              w=29.1241 + 1.01444E-5 * d,
                              a=0.387098,
                              e=0.205635 + 5.59E-10 * d,
                              M=168.6562 + 4.0923344368 * d,
                              d=d)
    self.Venus = OrbitalElements(N=76.6799 + 2.46590E-5 * d,
                            i=3.3946 + 2.75E-8 * d,
                            w=54.8910 + 1.38374E-5 * d,
                            a=0.723330,
                            e=0.006773 - 1.302E-9 * d,
                            M=48.0052 + 1.6021302244 * d,
                            d=d)
    self.Mars = OrbitalElements(N=49.5574 + 2.11081E-5 * d,
                          i=1.8497 - 1.78E-8 * d,
                          w=286.5016 + 2.92961E-5 * d,
                          a=1.523688,
                          e=0.093405 + 2.516E-9 * d,
                          M=18.6021 + 0.5240207766 * d,
                          d=d)
    self.Jupiter = OrbitalElements(N=100.4542 + 2.76854E-5 * d,
                              i=1.3030 - 1.557E-7 * d,
                              w=273.8777 + 1.64505E-5 * d,
                              a=5.20256,  # (AU)
                              e=0.048498 + 4.469E-9 * d,
                              M=19.8950 + 0.0830853001 * d,
                              d=d)
    self.Saturn = OrbitalElements(N=113.6634 + 2.38980E-5 * d,
                            i=2.4886 - 1.081E-7 * d,
                            w=339.3939 + 2.97661E-5 * d,
                            a=9.55475,
                            e=0.055546 - 9.499E-9 * d,
                            M=316.9670 + 0.0334442282 * d,
                            d=d)
    self.Uranus = OrbitalElements(N=74.0005 + 1.3978E-5 * d,
                            i=0.7733 + 1.9E-8 * d,
                            w=96.6612 + 3.0565E-5 * d,
                            a=19.18171 - 1.55E-8 * d,
                            e=0.047318 + 7.45E-9 * d,
                            M=142.5905 + 0.011725806 * d,
                            d=d)
    self.Neptune = OrbitalElements(N=131.7806 + 3.0173E-5 * d,
                              i=1.7700 - 2.55E-7 * d,
                              w=272.8461 - 6.027E-6 * d,
                              a=30.05826 + 3.313E-8 * d,
                              e=0.008606 + 2.15E-9 * d,
                              M=260.2471 + 0.005995147 * d,
                              d=d)
    print(self.__dict__)
      
def solarsystem(d):
  if isinstance(d, int):
    pass
  else:
    d = to_jd(d)
  
  return AstrologicalSolarSystem(d)

ss = solarsystem(datetime.now())

print(ss.Sun.compute_position())