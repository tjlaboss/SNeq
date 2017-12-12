# 22.211 - PSet06
#   by Travis J. Labossiere-Hickman (tjlaboss@mit.edu)
#
# Generic two-group one-dimensional diffusion solver


from scipy import *
import diffusion.nodes as nodes
from matplotlib import pyplot as plt

MAXINNER = 1000
MAXOUTER = 1100


def make_matrix(prob, G=2):
	m = int(prob.total_nx)
	n = m - 1
	L = zeros((m, m, G), dtype = float64)
	U = zeros((m, m, G), dtype = float64)
	b = zeros((m, m, G), dtype = float64)
	T = zeros((m, m), dtype = float64)
	
	for g in range(G):
		for i in range(1, n):
			ilist = [i - 1, i, i + 1]
			js = [None]*3
			
			for k in range(3):
				js[k] = prob.properties(ilist[k])
			
			ma = prob.mats[js[0]]
			dxa = prob.dxs[js[0]]
			
			mb = prob.mats[js[1]]
			dxb = prob.dxs[js[1]]
			
			mc = prob.mats[js[2]]
			dxc = prob.dxs[js[2]]
			
			(l, c, r), slist = nodes.interior_node(ma, mb, mc, dxa, dxb, dxc, g)
			L[i, i, g] = c
			L[i, i - 1, g] = l
			U[i, i + 1, g] = r
			if g == 0:
				# fast flux
				b[i, i, :] = slist
			elif g == 1:
				# thermal flux
				T[i, i] = slist[0]
			else:
				raise ValueError("This function works for 2 groups only, got " + str(G))
		
		# Boundary conditions
		ilist = [0, 1, 2, n-2, n-1, n]
		num = len(ilist)
		js = [None]*num
		for k in range(num):
			js[k] = prob.properties(ilist[k])
		
		ms = [None]*num
		dxs = [None]*num
		for k in range(num):
			j = js[k]
			ms[k] = prob.mats[j]
			dxs[k] = prob.dxs[j]
		
		if prob.bcs[0] == "reflective":
			(c, l), slist = nodes.reflective(ms[0], dxs[0], g)
			L[0, 0, g] = c
			U[0, 1, g] = l
			if g == 0:
				# fast flux
				b[0, 0, :] = slist[:]
			elif g == 1:
				# thermal flux
				T[0, 0] = slist[0]
		elif prob.bcs[0] == "vacuum":
			# zero-incoming flux from the left edge
			(l, c, r), slist = nodes.interior_node(ms[0], ms[1], ms[2], dxs[0], dxs[1], dxs[2], g)
			L[0, 0, g] = c
			U[0, 1, g] = r
			if g == 0:
				# g+1 = 1, fast flux
				b[0, 0, :] = slist[:]
			elif g == 1:
				# g+1 = 2, thermal
				T[0, 0] = slist[0]
			else:
				raise ValueError("This function works for 2 groups only, got " + str(G))
		else:
			raise NotImplementedError(prob.bcs)
			(l, c, r), slist = nodes.interior_node(ms[0], ms[1], ms[2], dxs[0], dxs[1], dxs[2], g)
			L[0, 1, g] = r
			L[0, 0, g] = c/(1 + 4*ms[0].D[g]/dxs[0])
			if g == 0:
				# g+1 = 1, fast flux
				b[n, n, :] = slist[:]
			elif g == 1:
				# g+1 = 2, thermal
				T[n, n] = slist[0]
			else:
				raise ValueError("This function works for 2 groups only, got " + str(G))
		
		if prob.bcs[-1] == "reflective":
			# reflective, right edge
			(c, r), slist = nodes.reflective(ms[-1], dxs[-1], g)
			L[n, n - 1, g] = r
			L[n, n, g] = c
			if g == 0:
				# fast flux
				b[n, n, :] = slist[:]
			elif g == 1:
				# thermal flux
				T[n, n] = slist[0]
		elif prob.bcs[-1] == "vacuum":
			# zero-incoming flux from the right edge
			(l, c, r), slist = nodes.interior_node(ms[-3], ms[-2], ms[-1], dxs[-3], dxs[-2], dxs[-1], g)
			L[n, n - 1, g] = l
			L[n, n, g] = c
			if g == 0:
				# g+1 = 1, fast flux
				b[n, n, :] = slist[:]
			elif g == 1:
				# g+1 = 2, thermal
				T[n, n] = slist[0]
			else:
				raise ValueError("This function works for 2 groups only, got " + str(G))
		else:
			raise NotImplementedError(prob.bcs)
			(l, c, r), slist = nodes.interior_node(ms[-3], ms[-2], ms[-1], dxs[-3], dxs[-2], dxs[-1], g)
			L[n, n - 1, g] = l
			L[n, n, g] = c/(1 + 4*ms[-1].D[g]/dxs[-1])
			if g == 0:
				# g+1 = 1, fast flux
				b[n, n, :] = slist[:]
			elif g == 1:
				# g+1 = 2, thermal
				T[n, n] = slist[0]
			else:
				raise ValueError("This function works for 2 groups only, got " + str(G))
	
	return L, U, b, T


def _gauss_seidel(L, U, s, x, k):
	"""A Gauss-Seidel iterative solver
	
	Inputs:
		invL:	array; inverse of (lower-triangular square matrix, contains the diagonal)
		U:		array; upper-triangular square matrix
		b:		array; vector of the same length as L and U
		x:		array; solution vector of last iteration
		k:      float; eigenvalue from the last iteration
	
	Outputs:
		x:	array; solution vector
	"""
	n = len(x) - 1
	m = int(len(x)/2)
	
	# More efficient calculation. Write this and compare the two
	# We know that [A] is tridiagonal
	
	# Leftmost
	x[0] = (s[0]/k - U[0,1]*x[1])/L[0, 0]
	# Interior
	for i in range(1,m):
		x[i] = (s[i]/k - L[i,i-1]*x[i-1] - U[i,i+1]*x[i+1])/L[i,i]
	for i in range(m,n):
		x[i] = (s[i]/k - L[i,i-1]*x[i-1] - U[i,i+1]*x[i+1] - L[i,i-m]*x[i-m])/L[i,i]
	# Rightmost
	x[n] = (s[n]/k - L[n,n-1]*x[n-1] - L[n,n-m]*x[n-m])/L[n,n]
	return x


def python_gauss_iterator(m, A, b,
                          eps_outer, eps_inner,
                          x=None, sguess=None, kguess=1.1):
	n = m*2
	if x is None:
		x = ones((n, 1))/n
	else:
		x.shape = (n, 1)
	l = tril(A)
	u = A - l
	
	# Guess the fission source.
	if sguess is None:
		sguess = b.dot(x)
	else:
		sguess.shape = (n, 1)
	
	fsdiff = 1
	kdiff = 1
	c = 0
	while ((fsdiff > eps_outer) or (kdiff > eps_outer)) and (c < MAXOUTER):
	#for z in range(50):
		fsdiff = 0
		oldx = x[:]
		c += 1
		
		# Converge flux at source term.
		fluxdiff = 1
		d = 0
		while fluxdiff > eps_inner and (d < MAXINNER):
			fluxdiff = 0
			d+=1
			
			# Do the math. Get the flux at this source term.
			x = _gauss_seidel(l, u, sguess, oldx, kguess)
			
			for i in range(n):
				fluxdiff += (x[i] - oldx[i])**2
			fluxdiff = sqrt(fluxdiff/n)
			oldx = x[:]
			
			if d >= MAXINNER:
				raise SystemError("Maximum inner iterations reached")
		
		# Debug
		# The inner iteration seems to be working just fine
		if c >= MAXOUTER:
			print("c =", c)
			print("k =", kguess)
			raise SystemError("Maximum number of iterations reached.")
		
		# Now the fission source has converged.
		# We know the flux 'x' at that fission source.
		# Find a new 'k' and guess
		s = b.dot(x)
		k = kguess * sum(s)/sum(sguess)
		kdiff = abs(k-kguess)/kguess
		kguess = k
		
		for i in range(m):
			fsdiff += (s[i] - sguess[i])**2
		fsdiff = sqrt(fsdiff/m)
		sguess = s[:]
	
	print("CMFD converged after", c, "fission source iterations.")
	
	print("Python k = ", kguess)
	
	return x, s, k


def solve_problem(prob, fluxguess=None, fsguess=None, kguess=None,
                  eps_outer=1E-5, eps_inner=1E-7, plot=False):
	"""Solve one of the 10 reactor physics problems
	
	Inputs:
		:param prob:        instance of problems.Problem;
							Should probably be in {prob0, prob 1, ..., prob9}
		:param plot:        Boolean; whether to plot the solution
							[Default: False]
		:param prob_num:    str or int; name or ID of the problem
							[Default: None]
	
	Outputs:
		[nothing]
	"""
	l, u, b, t2 = make_matrix(prob)
	m = len(l)
	n = 2*m
	
	# Build the big A matrix
	matA = zeros((n, n))
	matA[:m, :m] = (l + u)[:, :, 0]
	matA[m:, m:] = (l + u)[:, :, 1]
	matA[m:, :m] = -t2[:, :]
	
	matB = zeros((n, n))
	matB[:m, :m] = b[:, :, 0]
	matB[:m, m:] = b[:, :, 1]
	
	'''
	# Spy: check on the matrix shapes
	#plt.figure(); plt.spy(matA); plt.title("[A] matrix"); plt.show()
	#plt.figure(); plt.spy(matB); plt.title("[B] matrix"); plt.show()
	
	# Run either this or the Fortran...
	#fortran_gauss_iterator(m, matA, matB, plot)
	phi, s, k = python_gauss_iterator(m, matA, matB)
	
	
	phi /= phi[:m].mean()
	s /= s[:m].mean()
	sm = s.max()
	xpeaks = []
	# Find the location of the peaks
	for i, j in enumerate(s):
		if j == sm:
			# We're in a peak
			xpeaks.append(float(prob.distance(i)))
	print("Fission Source peaking of {:.3} occurs at: {:3.4} cm".format(sm, *xpeaks))
	'''
	return python_gauss_iterator(m, matA, matB, eps_outer, eps_inner,
	                             x=fluxguess, sguess=fsguess, kguess=kguess)
	
	
	if plot:
		xvals = [prob.distance(i) for i in range(m)]
		fluxvals = phi[:m] + phi[m:]
		titstr = "Python gauss-seidel"
		if prob_num is not None:
			titstr += " Problem " + str(prob_num)
		plt.suptitle(titstr)
		fscol = "orange"
		
		host = plt.subplot(111)
		host.plot(xvals, phi[:m], 'b-', label = "fast")
		host.plot(xvals, phi[m:], 'r-', label = "thermal")
		host.set_ylim([0, max(fluxvals)*1.02])
		host.plot([0], [-1], fscol, label= "Fission source")
		host.set_ylabel("$\phi(x)$", fontweight="bold", fontsize = 12)
		host.set_xlabel("$x$ (cm)", fontsize = 12)
		
		
		guest = plt.twinx()
		guest.plot(xvals, s[:m], fscol, label = "Fission source")
		guest.tick_params('y', colors =fscol)
		guest.set_ylabel("Fission Source", color=fscol, fontsize = 12)
		guest.set_ylim([0, sm*1.02])
		
		host.legend(loc = "center")
		plt.xlim([0, max(xvals)])
	
