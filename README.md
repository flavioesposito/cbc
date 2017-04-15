# cbc
Enumeration of Maximal Clique-Bicliques


# Abstract

"Find groups of close friends who like Bach, play violin and has visited every country whose name starts with "A" from a social network of users with additional information about interests of every user.
List of groups like the above can be found by enumerating all maximal clique-bicliques (CBCs) in a "join" of two networks: (a) social network (b) user-interest association. Each maximal CBC consists of a set of interests I and a group of users U such that (i) any two users in U are connected and (ii) every user in U is associated to every interest in I.

# Code

Download python library implementing relevant algorithms CBCGraphPKDD.py.

Download script for running algorithms: mcecbc.py.

Prints statistics of all maximal CBCs: python mcecbc.py interest-graph-file social-network-file
Write all maximal CBCs in an outfile: python mcecbc.py interest-graph-file social-network-file outfile
Print statistics of all maximal CBCs of sizes within a range; Imin & Imax: minimum & maximum size of interest set, Umin & Umax: minimum & maximum number of users in U, use -1 to not give any bound: python mcecbc.py interest-graph-file social-network-file Imin Imax Umin Umax
Write all maximal CBCs of sizes within a range in an outfile: python mcecbc.py interest-graph-file social-network-file outfile Imin Imax Umin Umax
Input Graph Files

Interest graph: There is one line for each edge. Each line has the format:u TAB v representing edge between interest u and user v
Social network graph: There is one line for each edge. Each line has the format:v TAB w representing edge between users v and w
Datasets

Sample for test: test.bigraph (interest graph) and test.graph (social graph).
Marker-Cafe: TheMarker.bigraph (interest graph) and TheMarker.graph (social graph).
Ning's creator network: Ning.bigraph (interest graph) and Ning.graph (social graph).
Ciao-DVD: rating.bigraph (interest graph) and trusts.graph (social graph) (social links are considered as undirected).
Filmtrust: ratings.bigraph (interest graph) and trust-uniq.graph (social graph) (social links are considered as undirected and without self-loops).
Erdos-Renyi random graphs: Random networks were generated, 10 of each kind, for different values of n (number of interests), m (number of users), p (probability of user-interest association) and q (probability of user-user association). Download tarball with varying m, varying n, varying p, varying q (scripts to generate the datasets:gen.py (main script), graphgen.py (library)). The filenames of the random graphs have the structure n-m-p-q-num (num goes from 0 to 9) and indicate the values used for generating those graphs.
