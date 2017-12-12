# Matrix builder


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
	center = Dab + Dcb + (mb.sigma_s12[g] + mb.sigma_a[g])*dxb
	if g == 0:
		sourcefrom1 = mb.nu_sigma_f[0]*dxb
		sourcefrom2 = mb.nu_sigma_f[1]*dxb
	else:
		sourcefrom1 = mb.sigma_s12[0]*dxb
		sourcefrom2 = 0
	
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
	center = 2*m.D[g]/dx + (m.sigma_s12[g] + m.sigma_a[g])*dx
	if g == 0:
		sourcefrom1 = m.nu_sigma_f[0]*dx
		sourcefrom2 = m.nu_sigma_f[1]*dx
	else:
		sourcefrom1 = m.sigma_s12[0]*dx
		sourcefrom2 = 0
	
	return [center, edge], [sourcefrom1, sourcefrom2]
