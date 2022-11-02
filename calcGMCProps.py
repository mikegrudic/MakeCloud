'''
Simple routines to calculate the cloud's: radius, 
average density, surface density, free-fall time (in years),
and number of particles (N) given a cloud mass and cell mass
resolution to choose 

Written by: Anna Rosen, 11/2/22
'''


import math
import numpy

MSUN = 1.989e33 #g
PCCM = 3.086e18 #cm
G = 6.6743e-8 #cm^3 g^-1 s
SECYR = 3600*24*365.25 #s

def calcRhoAve(Mcl, Rcl = None, Sigma = 1):
	'''
	Inputs:
	#cloud mass = Mcl [Msun]
	#cloud radius = Rcl [pc]
	#Sigma = Cloud surface density, default = 1 g/cm^2

	Output:
	rho_ave [g/cm^3]
	'''
	if Rcl is None and Sigma > 0.0:
		print('No Rcl supplied, calc. Rcl assuming Sigma = %.2e g/cm^2' % Sigma)
		Rcl = calcRcl(Mcl, Sigma)
	else:
		print('Error: must supple Rcl (in pc) or Sigma > 0') 
	M = Mcl*MSUN
	R = Rcl*PCCM
	rho_ave = 3*M/(4*math.pi * R**3.)
	print('Average cloud density =  %.2e g/cm^3' % rho_ave)
	return(rho_ave)


def calcRcl(Mcl, Sigma):
	'''
	Inputs:
	#cloud mass = Mcl [Msun]
	#cloud surface density = Sigma [g/cm^2]

	Output:
	Rcl [pc]
	'''
	M = Mcl * MSUN
	Rcl = (M/(math.pi *Sigma))**0.5/PCCM
	print('Cloud Radius = %.2f pc' % Rcl)
	return(Rcl)

def calc_SurfaceDensity(Mcl, Rcl):
	'''
	Inputs:
	#cloud mass = Mcl [Msun]
	#cloud radius = Sigma [g/cm^2]

	Output:
	Sigma [g/cm^2]
	'''
	M = Mcl*MSUN
	R = Rcl*PCCM
	Sigma = M/(math.pi * R**3.)
	print('Cloud surface density =  %.2e g/cm^2' % Sigma)
	return(rho_ave)

def calc_tff(Mcl, Rcl = None, Sigma = 1):
	'''
	Inputs:
	#cloud mass = Mcl [Msun]
	#cloud surface density = Sigma [g/cm^2]

	Output:
	tff [yr]
	'''
	if Rcl is None and Sigma > 0.0:
		print('No Rcl supplied, calc. Rcl assuming Sigma = %.2e g/cm^2' % Sigma)
		Rcl = calcRcl(Mcl, Sigma)
	else:
		print('Error: must supple Rcl (in pc) or Sigma > 0')

	rho_ave = calcRhoAve(Mcl, Rcl=Rcl, Sigma = 1)

	tff = (3*math.pi/(32*G*rho_ave))**0.5/SECYR
	print('tff = %.2f yr' %tff)
	return(tff)

def calc_Npart(Mcl, massRes = 1e-3):
	'''
	Inputs:
	#cloud mass = Mcl [Msun]
	#mass/cell [Msun]

	Output:
	N [particles]
	'''

	N = Mcl/massRes
	print('For Mcl = %.2e Msun, Mcell = %.2e' % (Mcl, massRes))
	print('N = %.2e particles' % N)
	return N

