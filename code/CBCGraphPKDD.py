import collections
import random
import time
import itertools
import sys, os

__author__ = 'Debajyoti Bera and Flavio Esposito'

#verbose = True
verbose = False

class GraphError(Exception):
    pass

''' Clique-Biclique graph'''
class CBCGraph:

    def __init__(self, bigraph_file, graph_file):
        self.U2V = collections.defaultdict(set)  # edges from left to right (unused)
        self.V2U = collections.defaultdict(set)  # edges from right to left
        self.V2V = collections.defaultdict(set)  # edges within right
        self.UVedges = collections.defaultdict() # adj. matrix for edges between left and right (unused)
        self.VVedges = collections.defaultdict() # adj. matrix for edges within right (unused)

	# Enumeration may take a long time, so we need to interrupt in the middle
	self.kill = False # flag to interrupt enumeration if time crossed threshold
	self.start_exp = 0 # timestamp when enumeration starts
	self.time_threshold = 0 # after what time to stop enumeration

	self.__init_graph(bigraph_file, graph_file)

#######################################################################

    def _map_edge_UV(self, edge):
        u, v = edge
        if u is None or v is None:
            raise GraphError('an edge must connect two nodes')
        self.U2V[u].add(v)
        self.V2U[v].add(u)
        self.UVedges[edge] = True

    def _map_edge_VV(self, edge):
        u, v = edge
	if u == v:
	    #print "Please remove from Graph file: %d %d" % (u,v)
	    return
        if u is None or v is None:
            raise GraphError('an edge must connect two nodes')
        self.V2V[u].add(v)
        self.V2V[v].add(u)
        self.VVedges[edge] = True
        # TODO: nodes for this edge may not be in bipartite graph in dataset is not sanitised
        if u not in self.V2U:
            self.V2U[u] = set()
        if v not in self.V2U:
            self.V2U[v] = set()

    def add_edge_UV(self, edge):
        self._map_edge_UV(edge)

    def add_edge_VV(self, edge):
        self._map_edge_VV(edge)

    def __init_graph(self, bigraph_file, graph_file):
	# In case only a sample of vertices should be added from a large graph
	u_max = v_max = -1
	
	#t0 = time.time()
	if verbose: print "adding Bipartite graph from", bigraph_file
        with open(bigraph_file,'r') as graph:
            for line in graph:
                (a,b) = line.strip().split()
                edge = (int(a), int(b))
		if (u_max == -1 or v_max == -1) or (edge[0] <= u_max and edge[1] <= v_max):
		    self.add_edge_UV(edge)
		    #print "adding ", edge

	if verbose: print "adding Graph from", graph_file
        with open(graph_file,'r') as graph:
            for line in graph:
                (a,b) = line.strip().split()
                edge = (int(a), int(b))
		if (v_max == -1) or (edge[0] <= v_max and edge[1] <= v_max):
		    self.add_edge_VV(edge)
		    #print "adding ", edge
    
	print "Graph: %s, Bipartite graph: %s\nStats: |U| = %d, |V| = %d, Total vertices = %d, Graph edges = %d, Bigraph edges = %d, Total edges = %d" % (graph_file,
		bigraph_file, self.U_count, self.V_count, self.U_count + self.V_count, len(self.VVedges),
		len(self.UVedges), self.edge_count)
        #print('elapsed time:', time.time() - t0)

#######################################################################

    @property
    def edge_count(self):
        return len(self.UVedges)+len(self.VVedges)

    @property
    def U_count(self):
        return len(self.U2V)

    @property
    def V_count(self):
        return len(self.V2U)

    @property
    def U(self):
        """
        returns a set of all "left" nodes
        """
        return self.U2V.keys()

    @property
    def V(self):
        """
        returns a set of all "right" nodes
        """
        return self.V2U.keys()

    def __repr__(self):
        return '\n'.join([str(self.U2V), str(self.V2U), str(self.V2V)])

############################################################################

    # maximal biclique enumeration by DOI: 10.1186/1471-2105-15-110
    def MBEA(self):
	self.kill = False
	try:
	    for bc in self._MBEA(set(self.U), set(), set(self.V), set(),1):
		yield bc
	except RuntimeError as re:
	    self.kill = True
	    return

    def _MBEA(self, L,R,P,Q, indent):
        """
        L - a set of vertices in U that are common neighbors of vertices
            in R
        R - a set of vertices in V belonging to the current biclique
        P - a set of vertices in V that can be added to R
        Q - a set of vertices in V that have been previously added to R
        """
	if self.start_exp > 0 and ((time.time()-self.start_exp) > self.time_threshold):
	    self.kill = True
	if self.kill:
	    return

        indstr = ''.join(['   '*indent])
        if verbose: print "%s Tree node: L=%s, R=%s, P=%s Q=%s" % (indstr,L,R,P,Q)
        while P:  # len(P) > 0:
            x = P.pop()
            if verbose: print "%s +++ Processing x=%s, L=%s, R=%s, P=%s Q=%s" % (indstr,x,L,R,P,Q)
            # extend biclique
            R_prime = R.copy() | {x}
            L_copy = L.copy()
            L_prime = L_copy & self.V2U[x]
            if len(L_prime) == 0:
                continue
            # opt
            L_prime_dash = L_copy - L_prime
            C = set([x])
            # create new sets
            P_prime = set()
            Q_prime = set()
            # check maximality
            is_LR_maximal = True
            for v in Q:
                # checks whether L_prime is a subset of all adjacent nodes
                # of v in Q
                Nv = L_prime & self.V2U[v]
                if len(Nv) == len(L_prime):
                    is_LR_maximal = False
                    break
                elif Nv:  # len(Nv) > 0:
                    # some vertices in L_prime are not adjacent to v:
                    # keep vertices adjacent to some vertex in L_prime
                    Q_prime.add(v)
            if is_LR_maximal:
                P_prime = []
                for v in P:
                    # get the neighbors of v in L_prime
                    Nv = L_prime & self.V2U[v]
                    if len(Nv) == len(L_prime):
                        R_prime.add(v)
                        S = L_prime_dash & self.V2U[v]
                        if len(S) == 0:
                            C.add(v)
                    elif len(Nv) > 0:
                        # some vertices in L_prime are not adjacent to v:
                        # keep vertices adjacent to some vertex in L_prime
                        P_prime.append((v, len(Nv)))

                sorted(P_prime, key=lambda x:x[1])
                P_prime = set([x[0] for x in P_prime])

                if verbose: print indstr,"*** BC:",(L_prime, R_prime)  # report maximal biclique
                yield (L_prime, R_prime)
                if P_prime:  # len(P_prime) > 0:
                    for bc in self._MBEA(L_prime, R_prime, P_prime, Q_prime, indent+1):
                        yield bc
            # move x to former candidate set
            Q = Q | C
            # opt
            P = P - C
            #print indstr,"... Removing C=%s, new P=%s, Q=%s" % (C,P,Q)

############################################################################

    # maximal clique enumeration by Bron-Kerbosch algorithm + Tomita's pivoting rule
    def Tomita(self):
	self.kill = False
	try:
	    for c in self._Tomita(set(), set(self.V), set(), False, 1, self.V2V):
		if verbose: print "Clique: ",c
		yield c
	except RuntimeError as re:
	    self.kill = True
	    return


    def _Tomita(self, R, P, X, indent, EdgeMap):
	if self.start_exp > 0 and ((time.time()-self.start_exp) > self.time_threshold):
	    self.kill = True
	if self.kill:
	    return

        indstr = ''.join(['   '*indent])
        if verbose: print "%s Calling Tomita on R=%s, P=%s, X=%s" % (indstr, R,P,X)

	if len(P) == 0:
	    if len(R) == 0:# R = {}
		return
	    else:# R != {}
		if len(X) == 0:#X = {}
		    if verbose: print "%s Tomita found clique: %s" % (indstr, R)
		    yield R # if P U X is empty
		    return
		else:#X != {}
		    return # no-op

	# shortcut
        if len(R) == 0 and len(X) == 0 and len(P) == 1:
            if verbose: print "%s Quick clique found: %s" % (indstr, P)
            yield P
            return

        # choose u in P U X to maximize |P intersect neigh(u)|
        max_u = -1
        max_u_val = -1
        max_Nu = set()
        for u in P | X:
            Nu = P & EdgeMap[u]
            if len(Nu) > max_u_val:
                max_u = u
                max_u_val = len(Nu)
                max_Nu = Nu

        for v in P - max_Nu:
            R_prime = R | {v}
            P_prime = P & EdgeMap[v]
            X_prime = X & EdgeMap[v]
            for c in self._Tomita(R_prime, P_prime, X_prime, indent+1, EdgeMap):
                yield c
            P = P - {v}
            X = X | {v}

############################################################################
    
    ''' Bron-Kerbosch based CBC enumeration
        Make U a complete graph and enumerate all maximal cliques in it '''
    def _BKCBC_prepare(self):
        # prepare graph
        # remap id: V -> V, U -> max(V)*10+U
        # so, any edge with both nodes > max(V) are dummy
        base = 10*max(self.V)

        AllEdges = self.V2V.copy()
        AllV = set(self.V).copy()

        for v in self.V:
            # remap
            for u in self.V2U[v]:
                AllEdges[v].add(base+u)
                AllEdges[base+u].add(v)
        # add dummy U2U adges
        for u in self.U:
            for v in self.U:
                if u != v:
                    AllEdges[base+u].add(base+v)
                    AllEdges[base+v].add(base+u)
            AllV.add(base+u)

	return (AllV, AllEdges)

    def _BKCBC_run(self, AllV, AllEdges, rangeNumU, rangeNumV):
        maxV = max(self.V)
	base = 10*maxV
	maxU = base + max(self.U)

	try:
        # get all cliques
	    for clique in self._TomitaCBC(set(), set(AllV), set(), maxU, maxV, rangeNumU, rangeNumV, 1, AllEdges):
		# filter dummy cliques
		clique_U = set()
		clique_V = set()
		for x in clique:
		    if x < base:
			clique_V.add(x)
		    else:
			clique_U.add(x-base)
		if len(clique_U) > 0 and len(clique_V) > 0:
		    yield (clique_U, clique_V)
	except RuntimeError as re:
	    print "Error! ",re
	    self.kill = True
	    return

    def _TomitaCBC(self, R, P, X, maxU, maxV, rangeNumU, rangeNumV, indent, EdgeMap):
	if self.start_exp > 0 and ((time.time()-self.start_exp) > self.time_threshold):
	    self.kill = True
	if self.kill:
	    return

        indstr = ''.join(['   '*indent])
        if verbose: print "%s Calling Tomita on R=%s, P=%s, X=%s (maxV = %d)" % (indstr, R,P,X, maxV)

	# count number of vertices in U in R and in P
	numR_U = len([x for x in R if x > maxV])
	numP_U = len([x for x in P if x > maxV])
	numR_V = len(R) - numR_U
	numP_V = len(P) - numP_U
	# do not continue ...
	# if R & P has vertices only in V (no vertices from U)
	# or R & P has vertices only in U (no vertices from V)
	# Then final clique cannot contain both U and V vertices
	if (numR_U == 0 and numP_U == 0) or (numR_V == 0 and numP_V == 0):
	    return

	if rangeNumV is not None:
	    minNumV = rangeNumV[0]
	    maxNumV = rangeNumV[1]
	    # Do not continue if
	    #    |R & V| > maxNumV (R is only going to increase)
	    # or |R & V| + |P & V| < minNumV (no other vertex will be added)
	    if (maxNumV != -1 and numR_V > maxNumV) or (minNumV != -1 and numR_V + numP_V < minNumV):
		return

	if rangeNumU is not None:
	    minNumU = rangeNumU[0]
	    maxNumU = rangeNumU[1]
	    # Do not continue if
	    #    |R & U| > maxNumU (R is only going to increase)
	    # or |R & U| + |P & U| < minNumU (no other vertex will be added)
	    if (maxNumU != -1 and numR_U > maxNumU) or (minNumU != -1 and numR_U + numP_U < minNumU):
		return

	if len(P) == 0:
	    if len(R) == 0:# R = {}
		return
	    else:# R != {}
		if len(X) == 0:#X = {}
		    if verbose: print "%s Tomita found clique: %s" % (indstr, R)
		    yield R # if P U X is empty
		    return
		else:#X != {}
		    return # no-op

	# shortcut
        if len(R) == 0 and len(X) == 0 and len(P) == 1:
            if verbose: print "%s Quick clique found: %s" % (indstr, P)
            yield P
            return

        # choose u in P U X to maximize |P intersect neigh(u)|
        max_u = -1
        max_u_val = -1
        max_Nu = set()
        for u in P | X:
            Nu = P & EdgeMap[u]
            if len(Nu) > max_u_val:
                max_u = u
                max_u_val = len(Nu)
                max_Nu = Nu

        for v in P - max_Nu:
            R_prime = R | {v}
            P_prime = P & EdgeMap[v]
            X_prime = X & EdgeMap[v]
            for c in self._TomitaCBC(R_prime, P_prime, X_prime, maxU, maxV, rangeNumU, rangeNumV, indent+1, EdgeMap):
                yield c
            P = P - {v}
            X = X | {v}

###################################################################################

    @staticmethod
    def to_str(cbc):
	if type(cbc) is tuple:
	    (U,V) = cbc
	    # sort U
	    U_list = sorted(list(U))
	    V_list = sorted(list(V))
	    return str(U_list) + ':' + str(V_list)
	else:
	    U = cbc
	    U_list = sorted(list(U))
	    return str(U_list)

##################################################################################

    @staticmethod
    # exp = 'bkcbc' / 'mce' / 'mbce'
    def run(cbcgraph, exp, outfile):
	print '---------- Enumerating %s ----------' % (exp)

	count = 0
	time_exp = 0
	file_fp = None
	if outfile is not None and len(outfile) > 0:
	    file_fp = open(outfile, 'w')

	do_f = None
	if exp == 'mcbc':
	    do_f = cbcgraph.mCBC
	elif exp == 'mcbcbasic':
	    do_f = cbcgraph.mCBCbasic
	elif exp == 'bkcbc':
	    (allv, alledges) = cbcgraph._BKCBC_prepare()
	    def run_bkcbc() :
		for x in cbcgraph._BKCBC_run(allv, alledges, None, None):
		    yield x
	    do_f = run_bkcbc
	elif exp == 'mce':
	    do_f = cbcgraph.Tomita
	elif exp == 'mbce':
	    do_f = cbcgraph.MBEA
	else:
	    raise NameError(exp)

	t0 = time.time()
        cbcgraph.time_threshold = 36000
        cbcgraph.start_exp = t0

	for x in do_f():
	    #if True or verbose: print "### %s : %s" % (exp, cbcgraph.to_str(x))
	    if file_fp:
		file_fp.write(cbcgraph.to_str(x) + '\n')
	    count = count + 1
	time_exp = time.time() - t0
	if file_fp:
	    file_fp.close()
	return (time_exp, count, cbcgraph.kill)

    @staticmethod
    def runMCECBC(cbcgraph, rangeNumU, rangeNumV, outfile):
	count = 0
	time_exp = 0
	file_fp = None
	if outfile is not None and len(outfile) > 0:
	    file_fp = open(outfile, 'w')

	(allv, alledges) = cbcgraph._BKCBC_prepare()

	t0 = time.time()
        cbcgraph.time_threshold = 36000
        cbcgraph.start_exp = t0

	for (l,r) in cbcgraph._BKCBC_run(allv, alledges, rangeNumU, rangeNumV):
	    if rangeNumU != None:
		if (rangeNumU[0] != -1 and len(l) < rangeNumU[0]) or \
		   (rangeNumU[1] != -1 and len(l) > rangeNumU[1]):
		       continue
	    if rangeNumV != None:
		if (rangeNumV[0] != -1 and len(r) < rangeNumV[0]) or (rangeNumV[1] != -1 and len(l) > rangeNumV[1]):
		       continue

	    #if True or verbose:
	    #	print exp,
	    #	print ''.join(['#' for a in l]),
	    #	print ' : ',
	    #	print ''.join(['@' for a in r])
	    if file_fp:
		file_fp.write(cbcgraph.to_str((l,r)) + '\n')
	    count = count + 1
	time_exp = time.time() - t0
	if file_fp:
	    file_fp.close()
	return (time_exp, count, cbcgraph.kill)

##############################################################

if __name__ == '__main__':
    print "This is a class file. Please run mcecbc.py."
