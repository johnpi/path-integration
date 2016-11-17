import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class MemCell:

  def calc_Gs(self, gs, Vpre, Elo, Ehi):
    if Vpre < Elo:
      Gs = 0
    elif Vpre >= Ehi:
      Gs = 1
    else:
      Gs = gs * (Vpre - Elo) / (Ehi - Elo)
    return Gs

  def init_mem_cell(self):
    self.V1 = self.Vinit
    Vdiff = self.V2 - self.V1
    if Vdiff == 0: Vdiff = 1
    Vpre = self.V1
    Gs = self.calc_Gs(self.gs, Vpre, self.Elo, self.Ehi)
    self.V2 = self.Elo - (self.Gm * (self.Er - self.V1) * (self.Ehi - self.Elo)) / (Gs * (Vdiff))


  def update_mem_cell(self, Vdesired):
    Iapp = (Vdesired - self.V1) * 3.1459

    for i in range(1, self.tsteps):
      #if i > 50: Iapp = 0
      #if i >= 100 and i <= 200: Iapp = (Vdesired - self.V1) * 3.1459
      if i >= 1 and i <= self.tsteps / 2: Iapp = (Vdesired - self.V1) * 3.1459
      else:       Iapp = 0
      
      Vpre = self.V2
      Gs = self.calc_Gs(self.gs, Vpre, self.Elo, self.Ehi)
      dV1 = (self.Gm * (self.Er - self.V1) + Gs * (self.Es - self.V1) + Iapp) * self.dt / self.Cm
      self.V1 += dV1

      Vpre = self.V1
      Gs = self.calc_Gs(self.gs, Vpre, self.Elo, self.Ehi)
      dV2 = (self.Gm * (self.Er - self.V2) + Gs * (self.Es - self.V2)) * self.dt / self.Cm
      self.V2 += dV2

      if self.debug:
	self.Iapp_t.append(Iapp)
	self.V1_t.append(self.V1)
	self.V2_t.append(self.V2)

  def print_values(self):
    print "V1 =", self.V1
    print "V2 =", self.V2

  def print_values_timeseries(self):
    print "V1(t) =", self.V1_t
    print "V2(t) =", self.V2_t

  def estimate_drift(self, tsteps = 1000):
    if tsteps > 0:
      tsteps = -1 * tsteps
    return self.V1_t[tsteps] - self.V1_t[-1]

  def plot_values(self):
    plt.figure(1)
    plt.subplot(311)
    plt.plot(self.Iapp_t, 'r-')
    plt.subplot(312)
    #plt.plot(V1_t, 'b-', dV1_t, 'b+')
    plt.plot(self.V1_t, 'b-')
    #plt.plot(dV1_t, 'k+')
    plt.subplot(313)
    #plt.plot(V2_t, 'r--', dV2_t, 'r+')
    plt.plot(self.V2_t, 'r--')
    #plt.plot(dV2_t, 'r+')
    plt.show()

  def get_value(self):
    # scale range to 0..1
    scaled0to1 = (self.V1 + 60.0) / 20.0
    return scaled0to1
    
  def set_value(self, speedScaled0to1):
    # Scale the input range 0..1 up to a range -60..-40
    Vdesired = speedScaled0to1 * 20.0 - 60.0
    self.update_mem_cell(Vdesired)
    
  def __init__(self,
               Cm = 1,     # uF
               Gm = 1,     #
               Er = -40,   #
               gs = 0.5,   #
               Elo = -60,  #
               Ehi = -40,  #
               dt = 0.1,   # Time step
               Es = -100,  # 
               Vinit = -50, # Initial memory Voltage value
               tsteps = 100, # it was 300,
               debug = False
               ):

    self.Cm = Cm
    self.Gm = Gm
    self.Er = Er
    self.gs = gs
    self.Elo = Elo
    self.Ehi = Ehi
    self.dt = dt
    self.Es = Es
    self.Vinit = Vinit
    self.tsteps = tsteps
    self.debug = debug

    self.V1 = self.Vinit
    self.V2 = 0

    # Result lists
    self.V1_t = []
    self.V2_t = []
    self.dV1_t = []
    self.dV2_t = []
    self.Iapp_t = []

    self.init_mem_cell()
    self.update_mem_cell(self.Vinit)
