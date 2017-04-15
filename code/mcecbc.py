import sys
from CBCGraphPKDD import CBCGraph

if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] == '-h' or sys.argv[1] == '--help':
	print 'python <scriptname.py> <bigraph.file> <graph.file> [outfile | <rangeU-min> <rangeU max> <rangeV-min> <rangeV-max>]'
	print 'Pass -u to python to stop output buffering.'
	sys.exit(1)
    bigraph_file = sys.argv[1]
    graph_file = sys.argv[2]
    outfile = None
    argv_index = 2
    if len(sys.argv) == 4 or len(sys.argv) == 8:
	outfile = sys.argv[3]
	argv_index = 3
    rangeNumU = rangeNumV = None
    if len(sys.argv) >= 7:
	rangeNumU = (int(sys.argv[argv_index+1]), int(sys.argv[argv_index+2]))
	rangeNumV = (int(sys.argv[argv_index+3]), int(sys.argv[argv_index+4]))

    cbcgraph = CBCGraph(bigraph_file, graph_file)

    total = 0
    num = 1 # set num if you want average time of multiple runs
    for i in range(num):
	(time_exp, count, killed) = CBCGraph.runMCECBC(cbcgraph, rangeNumU, rangeNumV, outfile)
	print "Experiment result for bounds L:%s and R:%s: Found %d in %f sec (Killed? %s)" % (rangeNumU, rangeNumV, count, time_exp, killed)
	total = total + time_exp
    print "=> Avg time (%d runs): %f" % (num, total*1.0/num)
