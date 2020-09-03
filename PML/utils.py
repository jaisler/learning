import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib import rc, cm
rc('text', usetex=True)

# -------------------------- #	
# Print matrix
def print_matrix(m, ext):
	M = assemble(m)
	Mv = M.M.values
	#plt.colorbar()
	plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
	plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
	plt.imshow(Mv, "seismic", vmax=0.001, vmin=-0.001)
	plt.savefig('mass_matrix_'+str(ext)+'.pdf', format='PDF')

# -------------------------- #	
# L2 norm
def errornorm(ue ,u, h, dt, p):	# L2 norm
	error = sqrt(assemble(dot(u - ue, u - ue) * dx))
	data = open('data.dat', 'r') # Abra o arquivo (leitura)
	inside = data.readlines()
	inside.append("%g %g %i %g\n" % (h, dt, p, error))   # insira seu conteúdo
	data = open('data.dat', 'w') # Abre novamente o arquivo (escrita)
	data.writelines(inside)    # escreva o conteúdo criado anteriormente nele.
	data.close()		

	print("h = %g, dt = %g, P = %i, error = %g" % (h, dt, p, error))	
	
	return error
