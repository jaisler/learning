from firedrake import *
import FIAT, finat

# -------------------------- #	
# Spectral method - Gauss-Lobatto-Legendre rule
# 1D
def gauss_lobatto_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLobattoLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLobattoLegendrePointSet
    finat_qr = finat.quadrature.QuadratureRule
    return finat_qr(finat_ps(fiat_rule.get_points()), fiat_rule.get_weights())

# 3D
def gauss_lobatto_legendre_cube_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_lobatto_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_lobatto_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result

# -------------------------- #	
# Spectral method - Gauss-Legendre rule
# 1D
def gauss_legendre_line_rule(degree):
    fiat_make_rule = FIAT.quadrature.GaussLegendreQuadratureLineRule
    fiat_rule = fiat_make_rule(FIAT.ufc_simplex(1), degree + 1)
    finat_ps = finat.point_set.GaussLegendrePointSet
    finat_qr = finat.quadrature.QuadratureRule
    return finat_qr(finat_ps(fiat_rule.get_points()), fiat_rule.get_weights())

# 3D
def gauss_legendre_cube_rule(dimension, degree):
    make_tensor_rule = finat.quadrature.TensorProductQuadratureRule
    result = gauss_legendre_line_rule(degree)
    for _ in range(1, dimension):
        line_rule = gauss_legendre_line_rule(degree)
        result = make_tensor_rule([result, line_rule])
    return result

# -------------------------- #
# Quadrature rule
def quadrature_rule(quadrature_points, degree, dimension):
	''' Quadrature rule - Gauss-Lobatto-Legendre, Gauss-Legendre and Equi-spaced'''
	''' It works for 2D and 3D. '''
	if quadrature_points == 'GLL':
		qr_x = gauss_lobatto_legendre_cube_rule(dimension=dimension, degree=degree) 
		qr_s = gauss_lobatto_legendre_cube_rule(dimension=(dimension-1), degree=degree)
	elif quadrature_points == 'GL':
		qr_x = gauss_legendre_cube_rule(dimension=dimension, degree=degree) 
		qr_s = gauss_legendre_cube_rule(dimension=(dimension-1), degree=degree) 
	elif quadrature_points == 'Equi':
		qr_x = None
		qr_s = None
	return(qr_x, qr_s)

