from firedrake import *

# -------------------------- #
# Damping fucntions
def damping_functions(V, ps, x, x1, x2, a_pml, y, y1, y2, b_pml, z, z1, z2, c_pml):
	''' Damping functions for the perfect matched layer

	Reference: Kaltenbacher et al. (2013) - A modified and stable version of a perfectly matched 
	layer technique for the 3-d second order wave equation in time domain with an application 
	to aeroacoustics '''

	aux1 = Function(V)
	aux2 = Function(V)
	value_max = 50.0

	dimension = aux1.geometric_dimension()

	# Sigma X		
	sigma_max_x = value_max # Max damping
	aux1.interpolate(conditional(And((x >= x1 - a_pml), x < x1), ((abs(x-x1)**(ps))/(a_pml**(ps))) * sigma_max_x, 0.0))
	aux2.interpolate(conditional(And(x > x2, (x <= x2 + a_pml)), ((abs(x-x2)**(ps))/(a_pml**(ps))) * sigma_max_x, 0.0))
	sigma_x = Function(V).interpolate(aux1+aux2)

	# Sigma Y
	sigma_max_y = value_max # Max damping
	aux1.interpolate(conditional(And((y >= y1 - b_pml), y < y1), ((abs(y-y1)**(ps))/(b_pml**(ps))) * sigma_max_y, 0.0))
	aux2.interpolate(conditional(And(y > y2, (y <= y2 + b_pml)), ((abs(y-y2)**(ps))/(b_pml**(ps))) * sigma_max_y, 0.0))
	sigma_y = Function(V).interpolate(aux1+aux2)

	if dimension < 3:
		return (sigma_x, sigma_y)
	else:
		# Sigma Z
		sigma_max_z = value_max
		aux1.interpolate(conditional(And((z >= z1 - c_pml), z < z1), ((abs(z-z1)**(ps))/(c_pml**(ps))) * sigma_max_z, 0.0))
		aux2.interpolate(conditional(And(z > z2, (z <= z2 + c_pml)), ((abs(z-z2)**(ps))/(c_pml**(ps))) * sigma_max_z, 0.0))
		sigma_z = Function(V).interpolate(aux1+aux2)
		return (sigma_x, sigma_y, sigma_z)

# -------------------------- #
# Damping fucntions - 2D
def damping_matrices_2D(sigma_x, sigma_y):
	'''Damping matrices for a two-dimensional problem'''
	Gamma_1 = as_tensor([[sigma_x, 0.0], [0.0, sigma_y]])
	Gamma_2 = as_tensor([[sigma_x-sigma_y, 0.0], [0.0, sigma_y-sigma_x]])

	return (Gamma_1, Gamma_2)

# -------------------------- #
# Damping fucntions - 3D
def damping_matrices_3D(sigma_x, sigma_y, sigma_z):
	'''Damping matrices for a three-dimensional problem'''
	Gamma_1 = as_tensor([[sigma_x, 0.0, 0.0], [0.0, sigma_y, 0.0], [0.0, 0.0, sigma_z]])
	Gamma_2 = as_tensor([[sigma_x-sigma_y-sigma_z, 0.0, 0.0], [0.0, sigma_y-sigma_x-sigma_z, 0.0], [0.0, 0.0, sigma_z-sigma_x-sigma_y]])
	Gamma_3 = as_tensor([[sigma_y*sigma_z, 0.0, 0.0], [0.0, sigma_x*sigma_z, 0.0], [0.0, 0.0, sigma_x*sigma_y]])

	return (Gamma_1, Gamma_2, Gamma_3)

