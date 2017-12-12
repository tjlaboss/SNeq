# Matrix builder

from numpy import array

def interior_node(ma, mb, mc, dxa, dxb, dxc, g):
	"""Return the three terms for an interior interface node
	
	Inputs:
		:param ma:  Material on the left
		:param mb:  Material in the middle
		:param mc:  Material on the right
		:param dxa: mesh spacing (delta x) on the left
		:param dxb: mesh spacing in the middle
		:param dxc: mesh spacing on the right
		:param g:   energy group, must be in {1, 2}
	
	Output:
		:return: [left, center, right], [source1, source2]
	"""
	assert g in range(2), "g must be 0 (fast) or 1 (thermal)"
	Dab = 2*ma.D[g]*mb.D[g]/(ma.D[g]*dxb + mb.D[g]*dxa)
	Dcb = 2*mc.D[g]*mb.D[g]/(mc.D[g]*dxb + mb.D[g]*dxc)
	left = -Dab
	right = -Dcb
	center = Dab + Dcb + mb.sigma_r[g]*dxb
	if g == 0:
		sourcefrom1 = mb.nu_sigma_f[0]*dxb
		sourcefrom2 = mb.nu_sigma_f[1]*dxb
	else:
		sourcefrom1 = mb.scatter_matrix[1, 0]*dxb
		sourcefrom2 = mb.scatter_matrix[0, 1]*dxb
	
	return [left, center, right], [sourcefrom1, sourcefrom2]
 


def reflective(m, dx, g):
	"""Return the two terms for a reflective boundary node

	Inputs:
		:param m:   Material on the right
		:param dx:  mesh spacing on the right
		:param g:   energy group, must be in {1, 2}

	Output:
		:return: [center, edge], [sourcefrom1, sourcefrom2]
	"""
	assert g in range(2), "g must be 0 (fast) or 1 (thermal)"
	edge = -2*m.D[g]/dx
	center = 2*m.D[g]/dx + m.sigma_r[g]*dx
	if g == 0:
		sourcefrom1 = m.nu_sigma_f[0]*dx
		sourcefrom2 = m.nu_sigma_f[1]*dx
	else:
		sourcefrom1 = m.scatter_matrix[1, 0]*dx
		sourcefrom2 = m.scatter_matrix[0, 1]*dx
		#xvec = array([dx, dx])
		#sourcefrom1, sourcefrom2 = m.scatter_matrix.dot(xvec)
	
	return [center, edge], [sourcefrom1, sourcefrom2]
