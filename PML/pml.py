# The scalar second-order acoustic wave equation is solved. 
# Space discretization: Continuous and Discontinuous Galerkin method.
# Time integration: implicit and explicity Newmark time-stepping scheme
# Equispaced and spectral quadrature points
# Perfect matched layer - Kaltenbacher et al. (2013) - second-order formulation

from firedrake import *
from firedrake.petsc import PETSc
import math, os, sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc, cm
rc('text', usetex=True)

from manualDelta import *
from timesteppingScheme import *
from spectral import *
from pml_damping import *
from plot import *

set_log_level(ERROR)

# -------------------------- #	
# Global variables

# Space parameters
expo = 6	
n = 2**(expo)
Lx = 1.0
Ly = 1.0	
Lz = 1.0
h = Lx/n

# PML Coordinates
lx = 0.1 #lenth x
ly = 0.1 #lenth y
lz = 0.1 #lenth z
x1 = lx
x2 = Lx - lx
a_pml = lx
y1 = ly
y2 = Ly - ly
b_pml = ly
z1 = lz
z2 = Lz - lz
c_pml = lz

# Time parameters		
T = 0.800001  # final time
q = 0.2  # Courant number

# File: print
#outfile = File("field/acoustic_wave.pvd") 
#projected = File("field/proj_output.pvd", project_output=True)
projected = File("field/proj_output.pvd", target_degree=1, target_continuity=H1)

# File: Erro
#if os.path.isfile('data.dat') == False:
#	data = open('data.dat','w')
#	data.write('#h dt P erro\n')
#	data.close() 

# -------------------------- #
# Gaussian source 
def Gaussian(t, t0=0.02, sigma_t=50000.0, amp=100.0):

	return (-2.0*amp*sigma_t*(t-t0)*math.exp(-sigma_t*(t-t0)*(t-t0)))

# -------------------------- #
# Ricker wavelet
def Source(t, freq=40.0, amp=200.0):
	
	return (amp*(1.0-(1.0/2.0)*(2.0*pi*freq)*(2.0*pi*freq)*t*t)*math.exp((-1.0/4.0)*(2.0*pi*freq)*(2.0*pi*freq)*t*t))

# -------------------------- #
# Extract information from the receivers located 5% below the top of the domain
def receivers(u_sol, x_rec):

	if dimension == 2:
		points = [(x_, 0.5) for x_ in x_rec]  # 2D points
		u_rec = np.array([u_sol(point) for point in points])
	else:    
		points = [(x_, 0.5, 0.5) for x_ in x_rec]  # 3D points
		u_rec = np.array([u_sol(point) for point in points])

	return u_rec

# -------------------------- #
# Acoustic wave equation solver: Newmark scheme and PML
def acoustic():

	u = TrialFunction(V)
	u_prevs = [Function(V), Function(V)]
	v_i = Function(V)
	v = TestFunction(V) # Test Function

	# Initial condition
	u_prevs[0].assign(0.0)
	u_prevs[1].assign(0.0)   
	v_i.assign(0.0)   

	# Works for 2D - it should be extended to 3D
	(qr_x, qr_s) = quadrature_rule(quadrature_points, degree, dimension)

	# Mass matrix integrand
	m = u * v * dx(rule=qr_x) 
	if space == 'CG':
		# Stiffness matrix integrand
		a = c * c * dot(grad(v), grad(u)) * dx(rule=qr_x)
	elif space == 'IP-DG':
		# A unit normal vector that can be used in integrals over exterior and interior facets.
		n = FacetNormal(mesh)
		# Stiffness matrix integrand
		a =  (inner(c*c*grad(v),grad(u))*dx(rule=qr_x) 
			 + S*(inner(jump(u,n),avg(c*c*grad(v))))*dS(rule=qr_s) # Interior boundary
			 - (inner(jump(v,n),avg(c*c*grad(u))))*dS(rule=qr_s) # Interior boundary
			 + penalty*(avg(c*c)*inner(jump(u,n),jump(v,n)))*dS(rule=qr_s)) # Interior boundary: penalty
			 #+ S*dot(c*c*grad(v), n*u)*ds(rule=qr_s) # External boundary
			 #- dot(c*c*grad(u), n*v)*ds(rule=qr_s) # External boundary 
			 #+ penalty*u*v*ds(rule=qr_s)) # Exterior boundary: penalty

	# Non-reflective boundary condition - (Neumann boundary condition)
	nf = - c * ((u - u_prevs[1])/dt) * v * ds(rule=qr_s)

	# current time
	t = 0.0 

	# Source position
	if dimension == 2:
		#delta = Function(V).interpolate(Delta(eps=1e-4, x0=np.array([0.5, 0.5]), degree=5))
		x0=np.array([0.5, 0.5])
		delta = spatialDelta(V, x0, dimension, x, y)
	else:	
		#delta = Function(V).interpolate(Delta(eps=1e-4, x0=np.array([0.5, 0.5, 0.5]), degree=5))
		x0=np.array([0.5, 0.5, 0.5])		
		delta = spatialDelta(V, x0 , dimension, x, y, z)

	# Time dependent source
	f0 = Function(V).assign(delta*Source(t))
	f1 = Function(V).assign(delta*Source(t+dt))
	f2 = Function(V)

	# Newmark time-stepping schme - initial condition (first step)
	(LHS, RHS_i) = Newmark_scheme_initial(beta, gamma, m, a, v, u_prevs, v_i, f0, f1, qr_x, dt)

	# Linear and bilinear parts with the non-reflective boundary condition
	F = - LHS + RHS_i + dt*dt*nf
	lhs_ = lhs(F)
	rhs_ = rhs(F)

	# Assemble - lhs
	A = assemble(lhs_)
	
	# Assemble - rhs
	b = assemble(rhs_)

	# Solver parameters
	params={'ksp_type': 'gmres'}

	# Solver
	u_ = Function(V, name="pressure")
	solver = LinearSolver(A, P=None, solver_parameters=params)	
	solver.solve(u_, b)			
	u_prevs[1].assign(u_)	    		
	t += dt

	# Receivers
	u_rec = []

	# Right hand side
	RHS = Newmark_scheme_update_rhs(beta, gamma, m, a, v, u_prevs, f0, f1, f2, qr_x, dt)

	# PML equations 
	if PML:
		p = TrialFunction(W) # Trial Finction
		p_prevs = Function(W)
		p_prevs.assign(as_vector((0.0, 0.0))) # Initial condition
		q = TestFunction(W) # Test Function

		# 2D formulation
		if dimension == 2:
	
			(sigma_x, sigma_y) = damping_functions(V, ps, x, x1, x2, a_pml, y, y1, y2, b_pml, z=None, z1=None, z2=None, c_pml=None) # damping functions
			(Gamma_1, Gamma_2) = damping_matrices_2D(sigma_x, sigma_y) # damping matrices
	
			# Vectorial equation
			# Integrand			
			g = dot(p, q) * dx(rule=qr_x) #Right-hand side
			l = (dot(p_prevs, q) * dx(rule=qr_x) 
				+ dt * c * c * inner(grad(u_prevs[1]), dot(Gamma_2, q)) * dx(rule=qr_x) 
				- dt * inner(dot(Gamma_1, p_prevs), q) * dx(rule=qr_x)) #Left-hand side
			G = assemble(g) # Assemble - left-hand side

			# Scalar equation
			# Integrand
			add_a = - (sigma_x + sigma_y) * ((u - u_prevs[1])/dt) * v * dx(rule=qr_x) - sigma_x * sigma_y * u_prevs[1] * v * dx(rule=qr_x)
			add_f = + inner(p_prevs, grad(v)) * dx(rule=qr_x)

			if space == 'CG':
				# Non-liear form
				F = - LHS + RHS + dt*dt*(add_a + add_f + nf)
		
			elif space == 'IP-DG':
				# Vectorial equation						
				add_dg_l = - inner(jump(u_prevs[1], n), avg(dot(Gamma_2, q))) * dS(rule=qr_s)
				l = l + dt*add_dg_l # update with dg term
				# Scalar equation
				add_dg_f = - inner(avg(p_prevs), jump(v,n)) * dS(rule=qr_s)
	
				# Non-liear form		
				F = - LHS + RHS + dt*dt*(add_a + add_f + nf + add_dg_f)
			
			# Linear and bilinear form			
			lhs_ = lhs(F)
			rhs_ = rhs(F)	

			# Assemble - rhs
			A = assemble(lhs_)
	
			# Solver - vectorial equation
			p = Function(W)
			solver_v = LinearSolver(G, P=None, solver_parameters=params)	
		
			# Solver - scalar equation
			solver_s = LinearSolver(A, P=None, solver_parameters=params)	
	
			j = 1
			for i in range(fstep-1):
				if i > dstep:
					f0.assign(0.0)
					f1.assign(0.0)
					f2.assign(0.0)					
				else:
					# Time dependent source
					f0.assign(delta*Source(t-dt))
					f1.assign(delta*Source(t))
					f2.assign(delta*Source(t+dt))
					
				# Solver - scalar equation		
				L = assemble(l)
				solver_v.solve(p, L)		
				p_prevs.assign(p)
		
				# Solver - scalar equation
				b = assemble(rhs_)	
				solver_s.solve(u_, b)		
				u_prevs[0].assign(u_prevs[1])
				u_prevs[1].assign(u_)	
				t += dt
				PETSc.Sys.Print('t=',t)

				j += 1
				if j == wstep:
					#outfile.write(u_)
					projected.write(u_)
					#plot(u_)
					u_rec.append(receivers(u_, x_rec))
					j = 0
			
			# Plot profile
			#profile(u_rec, x_rec)
			#plt.show()
		
		elif dimension == 3:	

			omega = TrialFunction(Z) # Trial Finction
			omega_prevs = Function(Z)
			omega_prevs.assign(0.0) # Initial condition
			theta = TestFunction(Z) # Test Function

			(sigma_x, sigma_y, sigma_z) = damping_functions(V, ps, x, x1, x2, a_pml, y, y1, y2, b_pml, z, z1, z2, c_pml) # damping functions
			(Gamma_1, Gamma_2, Gamma_3) = damping_matrices_3D(sigma_x, sigma_y, sigma_z) # damping matrices

			# Vectorial equation - (II)
			# Integrand			
			g = dot(p, q) * dx(rule=qr_x) #Right-hand side
			l = (dot(p_prevs, q) * dx(rule=qr_x) 
				- dt * inner(dot(Gamma_1, p_prevs), q) * dx(rule=qr_x)
				+ dt * c * c * inner(grad(u_prevs[1]), dot(Gamma_2, q)) * dx(rule=qr_x) 
				- dt * c * c * inner(grad(omega_prevs), dot(Gamma_3, q)) * dx(rule=qr_x)) #Left-hand side
			G = assemble(g) # Assemble - left-hand side
	
			# Scalar equation - (III)
			# Integrand
			o = omega * theta * dx
			d = omega_prevs * theta * dx + dt * u_prevs[1] * theta * dx
			O = assemble(o) # Assemble - left-hand side		

			# Scalar equation - main equation - (I)
			# Integrand
			add_a = (- (sigma_x + sigma_y + sigma_z) * ((u - u_prevs[1])/dt) * v * dx(rule=qr_x) 
				- (sigma_x * sigma_y + sigma_x * sigma_z + sigma_y * sigma_z) * u_prevs[1] * v * dx(rule=qr_x) 
				- (sigma_x * sigma_y * sigma_z) * omega_prevs * v * dx(rule=qr_x)) 
			add_f = + inner(p_prevs, grad(v)) * dx(rule=qr_x)
	
			if space == 'CG':
				# Non-liear form - main equation - (I)
				F = - LHS + RHS + dt*dt*(add_a + add_f + nf)
		
			elif space == 'IP-DG':
				# Vectorial equation						
				add_dg_l = (- c*c*inner(jump(u_prevs[1], n), avg(dot(Gamma_2, q))) * dS(rule=qr_s) 
					    + c*c*inner(jump(omega_prevs, n), avg(dot(Gamma_3, q))) * dS(rule=qr_s))
				l = l + dt*add_dg_l # update with the dg term - (II)
				# Scalar equation
				add_dg_f = - inner(avg(p_prevs), jump(v,n)) * dS(rule=qr_s)
	
				# Non-liear form		
				F = - LHS + RHS + dt*dt*(add_a + add_f + nf + add_dg_f)

			# Linear and bilinear form			
			lhs_ = lhs(F)
			rhs_ = rhs(F)	

			# Print - Mass matrix
			#print_matrix(lhs_, 'I')
			#print_matrix(g, 'II')
			#print_matrix(o, 'III')

			# Assemble - rhs
			A = assemble(lhs_)
	
			# Solver - vectorial equation - (II)
			p = Function(W)
			solver_v = LinearSolver(G, P=None, solver_parameters=params)			
		
			# Solver - scalar equation - (III)
			omega = Function(Z)
			solver_o = LinearSolver(O, P=None, solver_parameters=params)	

			# Solver - scalar equation - main equation - (I)
			solver_s = LinearSolver(A, P=None, solver_parameters=params)	

			j = 1
			for i in range(fstep-1):
				if i > dstep:
					f0.assign(0.0)
					f1.assign(0.0)
					f2.assign(0.0)					
				else:
					# Time dependent source
					f0.assign(delta*Source(t-dt))
					f1.assign(delta*Source(t))
					f2.assign(delta*Source(t+dt))
	
				# Solver - vectorial equation - (II)
				L = assemble(l)
				solver_v.solve(p, L)		
				p_prevs.assign(p)

				# Solver - scalar equation - (III)
				D = assemble(d)
				solver_o.solve(omega, D)
				omega_prevs.assign(omega)
		
				# Solver - scalar equation - main equation - (I)
				b = assemble(rhs_)	
				solver_s.solve(u_, b)		
				u_prevs[0].assign(u_prevs[1])
				u_prevs[1].assign(u_)	
				t += dt
				PETSc.Sys.Print('t=',t)

				j += 1
				if j == wstep:
					outfile.write(u_)
					u_rec.append(receivers(u_, x_rec))
					j = 0
			
			# Plot profile
			profile(u_rec, x_rec)
			
	# ----------------------------------- #
	# without PML
	else:

		F = - LHS + RHS + dt*dt*nf
		lhs_ = lhs(F)
		rhs_ = rhs(F)

		# Assemble - lhs
		A = assemble(lhs_)
	
		# Solver
		solver_s = LinearSolver(A, P=None, solver_parameters=params)	

		j = 1
		for i in range(fstep-1):
			if i > dstep:
				f0.assign(0.0)
				f1.assign(0.0)
				f2.assign(0.0)					
			else:
				# Time dependent source
				f0.assign(delta*Source(t-dt))
				f1.assign(delta*Source(t))
				f2.assign(delta*Source(t+dt))
	
			# Solver - scalar equation
			b = assemble(rhs_)
			solver_s.solve(u_, b)		
			u_prevs[0].assign(u_prevs[1])
			u_prevs[1].assign(u_)	
			t += dt
			PETSc.Sys.Print('t=',t)

			j += 1
			if j == wstep:
				#outfile.write(u_)
				#projected.write(u_)
				#plot(u_)
				u_rec.append(receivers(u_, x_rec))
				j = 0
			
		# Plot profile
		profile(u_rec, x_rec)
		plt.show()

	# Plot solution
	#plt.plot(u_)
	#plt.show()

	# Calcular a normal L2 do erro
	#ue = Function(V).interpolate(t*t*sin(pi*x)*sin(pi*y))
	#return(errornorm(ue, u, h, dt, degree))

	return(u_rec)

# ----------- Main Program --------------- #			

if __name__ == "__main__":

	PETSc.Sys.Print('Setting up mesh across %d processes' % COMM_WORLD.size)

	elements = ['tria'] # elements tria, quad, tetra
	method = ['CG']  # space discretization 
	if method[0] == 'IP-DG' and elements[0] == 'quad':
		dg_space = 'DQ'
	elif method[0] == 'IP-DG':
		dg_space = 'DG'
	dimension = 2 # dimension
	quadrature_points = 'Equi' # quadrature points - Equi, GLL, GL
	degree = 2 # polynomial order

	#Newmark coefficients
	beta = 0.25
	gamma = 0.5

	# Acoustic equations with PML
	PML = True # True or False

	for el in elements:
		# Mesh definition
		if dimension == 2:
			if el == 'quad':
				mesh = RectangleMesh(n, n, Lx, Ly, quadrilateral=True)
				PETSc.Sys.Print("Quadrilateral elements")
			elif el == 'tria':
				mesh = RectangleMesh(n, n, Lx, Ly, quadrilateral=False)
				PETSc.Sys.Print("Triangular elements")
			PETSc.Sys.Print('Two-dimensional mesh')
			# Coordinates
			x, y = SpatialCoordinate(mesh) 
		elif dimension == 3:
			if el == 'tetra':
				mesh = BoxMesh(n, n, n, Lx, Ly, Lz)
				PETSc.Sys.Print("Tetrahedral elements")
			# Coordinates
			x, y, z = SpatialCoordinate(mesh) 
			PETSc.Sys.Print('Three-dimensional mesh')

		# Method: Continuous Galerkin or Discontinuous Galerkin
		for space in method:
			if space == 'CG':	
				PETSc.Sys.Print('Continuous Galerkin method')
				# CG - Continuous Galerkin
				if quadrature_points == 'GLL' or quadrature_points == 'GL':
					element = FiniteElement(space, mesh.ufl_cell(), degree=degree, variant='spectral') 
					PETSc.Sys.Print("Spectral quadrature points")
				else:
					element = FiniteElement(space, mesh.ufl_cell(), degree=degree, variant='equispaced') 
					PETSc.Sys.Print("Equi-spaced quadrature points")
			elif space == 'IP-DG':	
				PETSc.Sys.Print('Interior Penalty Discontinuous Galerkin method')
				# IP-DG - Interior penalty discontinuous Galerkin
				# Incomplete Interior Penalty Galerkin (IIPG)
				#S = 0.0
				# Symmetric Interior Penalty Galerkin (SIPG)
				S = -1.0
				# Non-Symmetric Interior Penalty Galerkin (NIPG)
				#S = 1.0
				
				if quadrature_points == 'GLL' or quadrature_points == 'GL':
					element = FiniteElement(dg_space, mesh.ufl_cell(), degree=degree, variant='spectral') 					
					PETSc.Sys.Print("Spectral quadrature points")
				else:
					element = FiniteElement(dg_space, mesh.ufl_cell(), degree=degree, variant='equispaced') 					
					PETSc.Sys.Print("Equi-spaced quadrature points")
				
				# Penalty function
				if dimension == 2:
					if el == 'quad':
						penalty = ((degree+1.0)*(degree+2.0))/(2.0*h) # De Basabe 2010 - Quads
					elif el == 'tria':
						penalty = ((degree+1.0)*(degree+2.0)*(2.0+sqrt(2.0)))/h # Shahbazi 2004 - tri
				elif dimension == 3:
					if el == 'tetra':
						diam = h/2.0 # diameter of the inscribed sphere of regular tetrahedon
						C = 1.0 # positive parameter
						penalty = C*(degree*(degree+2.0))/diam # Geevers 2018 - tetrahedral - spectral			
					
			# Space - Scalar
			V = FunctionSpace(mesh, element) 					
			# PML
			if PML:
				# Space - Vectorial
				W = VectorFunctionSpace(mesh, element) # Space
				if dimension == 3:
					Z = FunctionSpace(mesh, element) # Space

			# Wave speed
			c = Function(V).assign(1.0)

			cmax = 1.0
			dt = (q*h)/cmax # timestep size
			PETSc.Sys.Print('dt= ', dt)
			fstep = int(T/dt) 
			dstep = int(0.075/dt) # source time
			wstep = int(0.1/dt) # write the profile of the wave

    		# Configure receivers positions
			num_receivers = 200
			tol = 0.001  # avoid hitting points outside the domain
			x_rec = np.linspace(0.0 + tol, 1.0 - tol, num_receivers) # x-axis

			rec_all = []
			for ps in range(4):
				# Acoustic solver
				PETSc.Sys.Print('Calculating for m=', ps, '...')
				rec_all.append(acoustic())

			plot_all(rec_all, x_rec)



