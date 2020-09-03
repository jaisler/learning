from firedrake import *

# -------------------------- #
# Dirac delta 
class Delta(Expression):
	def __init__(self, eps, x0, **kwargs):
		Expression.__init__(self, **kwargs)
		self.eps = eps
		self.x0 = x0 

	def eval(self, value, x):
		eps = self.eps
		value[0] = eps/pi/(np.linalg.norm(x-self.x0)**2 + eps**2)

# -------------------------- #
# Dirac delta 
def spatialDelta(V, x0, dimension, x, y, z=None, sigma_x=2000.0):

	spdelta = Function(V)
	if dimension == 2:
		spdelta.interpolate(exp(-sigma_x*((x-x0[0])*(x-x0[0])+(y-x0[1])*(y-x0[1]))))
	else:  
		spdelta.interpolate(exp(-sigma_x*((x-x0[0])*(x-x0[0])+(y-x0[1])*(y-x0[1])+(z-x0[2])*(z-x0[2]))))
	
	return spdelta
