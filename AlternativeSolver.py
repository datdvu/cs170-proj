import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
from student_utils_sp18 import *
import itertools

from copy import copy, deepcopy
import sys
import heapq
import itertools
import solver_template
import solver0


from copy import copy, deepcopy


def travellingcost(matrix, path, kingdomNames):
	cost = 0
	for i in range(len(path) - 1):
		cost = cost + matrix[kingdomNames.index(path[i])][kingdomNames.index(path[i+1])]
	return cost

def totalcost(matrix, conqueredList, path, kingdomNames):
	cost = 0
	for i in range(len(path) - 1):
		cost = cost + matrix[kingdomNames.index(path[i])][kingdomNames.index(path[i+1])]
	for x in conqueredList:
		cost = cost + matrix[kingdomNames.index(x)][kingdomNames.index(x)]
	return cost

def findMinTourAlternative3(startingKingdomIndex, cqList, matrix, maxtrix_size, kingdomNames):
	"""
	Use Dijkstra to find the shortest path to the most "effective" vertex in the list of vertices to conquer.
	
	Remove that vertex and set that vertex to be the next starting point for Dijkstra.

	Keep removing until the list of kingdoms to conquer is empty.

	"""
	
	kingdoms_to_conquer = set(cqList)
	if (kingdomNames[startingKingdomIndex] in kingdoms_to_conquer):
		kingdoms_to_conquer.remove(kingdomNames[startingKingdomIndex])
	currentKingdom = startingKingdomIndex
	nextKingdom = startingKingdomIndex
	path = []
	g = Graph(maxtrix_size)
	while(len(kingdoms_to_conquer) != 0):
		kingdoms_in_conquered_list = []
		best_path = []
		for x in kingdoms_to_conquer:
			dijkstrapath = g.dijkstra(matrix, kingdomNames, currentKingdom, kingdomNames.index(x))
			z = []
			for y in dijkstrapath:
				if y in kingdoms_to_conquer and y != x:
					z.append(y)
			if len(z) >= len(kingdoms_in_conquered_list):
				kingdoms_in_conquered_list = z
				nextKingdom = kingdomNames.index(x)
				best_path = dijkstrapath
		for k in kingdoms_in_conquered_list:
			kingdoms_to_conquer.remove(k)
		kingdoms_to_conquer.remove(kingdomNames[nextKingdom])
		path = path + [kingdomNames[currentKingdom]] + best_path[:-1]
		currentKingdom = nextKingdom
	path = path + [kingdomNames[currentKingdom]]
	if path[-1] != kingdomNames[startingKingdomIndex]:
		path += g.dijkstra(matrix, kingdomNames, kingdomNames.index(path[-1]), startingKingdomIndex)
	return path

def TSP(matrix, list_of_kingdoms_to_visit, startingKingdomIndex, endingKingdomIndex):
	"""
	Return the min cost to visit all list of kingdoms starting from starting Kingdom and end at endingKingdom
	

	[1,{2,3,4}] [,1] C12 + C23 + C34 + C14


	[2,{3,4}] [,2] C23 + C34 + C14

	[3, {4}] [,3] C34+C14

	[4, {}] [4] C14
	"""
	if (len(list_of_kingdoms_to_visit) == 0):
		return ([endingKingdomIndex], matrix[startingKingdomIndex][endingKingdomIndex])
	possibleSolutions = {}
	for x in list_of_kingdoms_to_visit:
		newListToVisit = set(list_of_kingdoms_to_visit)
		newListToVisit.remove(x)
		sol = TSP(matrix, newListToVisit, startingKingdomIndex, x)
		possibleSolutions[sol[1]] = (x,sol[0])
	minSol = possibleSolutions[min(possibleSolutions.keys())]
	return (minSol[1] + [endingKingdomIndex], min(possibleSolutions.keys()) + matrix[minSol[0]][endingKingdomIndex])

def findMinTour(startingKingdomIndex, fullWalk, cqList, matrix, maxtrix_size, kingdomNames):
	"""
	Return the min cost to "visit" all vertices of the full Walk 
	and conquer all kingdoms in conquered list
	fullWalk_1 = ['a1','a2', 'a1', 'a4', 'a3', 'a4', 'a5','a4','a1']
	fullWalk_2 = ['a1','a2','a6','a2','a1','a4','a5','a4','a3','a4','a1']
	cqList_1 = ['a1','a4']
	cqList1_1 = ['a2', 'a4']
	cqList_2 = ['a1','a4','a6']

	matrix_1 = [[0,20,10,8,0,0],
				[20,0,20,0,20,5],
				[10,20,0,5,0,0],
				[8,0,5,0,3,0],
				[0,20,0,3,0,0],
				[0,5,3,0,0,0]]
	
	kingdomNames1 = ['a' + str(i) for i in range(1, 6)]
	kingdomNames2 = ['a' + str(i) for i in range(1, 7)]
	
	findMinTour(0, fullWalk_1, cqList_1, matrix_1, len(matrix_1), kingdomNames1) == ['a1', 'a4']
	findMinTour(0, fullWalk_1, cqList1_1, matrix_1, len(matrix_1), kingdomNames1) == ['a1', 'a2', 'a1', 'a4']
	findMinTour(0, fullWalk_2, cqList_2, matrix_1, len(matrix_1), kingdomNames2) == ['a1', 'a2', 'a6', 'a2', 'a1', 'a4']


	"""

	# define a path 

	returned_path = []
	temp_path = []
	conquered = []
	lastKingdom = 0

	for k in range(len(fullWalk)):
		v = fullWalk[k]
		if k == 0:
			returned_path.append(v)
			if v in cqList:
				conquered.append(v)
				if (len(conquered) == len(cqList)):
						break
			temp_path = [v]
		else:
			if v in cqList:
				if v not in conquered:
					conquered.append(v)
					returned_path.extend(temp_path[1:])
					returned_path.append(v)
					if (len(conquered) == len(cqList)):
						break
					temp_path = [v]
				else:
					if v == temp_path[0]:
						temp_path = [v]
					else:
						returned_path.extend(temp_path[1:])					
						returned_path.append(v)
						temp_path = [v]

			else:
				if v in temp_path:
					notEncountered = True
					i = 0
					index = 0
					while notEncountered and i < len(temp_path):
						if temp_path[i] == v:
							index = i
							notEncountered = False
						i = i + 1
					temp_path = temp_path[0:(index+1)]
				else:
					temp_path.append(v)
			#print(temp_path)
	#print(isSurrender)
	
	#print(temp_path)
	#print(isSurrender)

	for i in range(len(kingdomNames)):
		if kingdomNames[i] == returned_path[-1]:
			lastKingdom = i
			break
	#print("The last kingdom: ", lastKingdom)
	g = Graph(maxtrix_size)

	dijkstraPath = g.dijkstra(matrix, kingdomNames, lastKingdom, startingKingdomIndex)

	#print("Dijkstra from the last kingdom back to the starting one: ", dijkstraPath)
	returned_path += dijkstraPath
	return returned_path

def findMinTourAlternative(startingKingdomIndex, cqList, matrix, maxtrix_size, kingdomNames):
	"""
	Return the min cost to "visit" all vertices of the full Walk 
	and conquer all kingdoms in conquered list
	fullWalk_1 = ['a1','a2', 'a1', 'a4', 'a3', 'a4', 'a5','a4','a1']
	fullWalk_2 = ['a1','a2','a6','a2','a1','a4','a5','a4','a3','a4','a1']
	cqList_1 = ['a1','a4']
	cqList1_1 = ['a2', 'a4']
	cqList_2 = ['a1','a4','a6']

	matrix_1 = [[0,20,10,8,0,0],
				[20,0,20,0,20,5],
				[10,20,0,5,0,0],
				[8,0,5,0,3,0],
				[0,20,0,3,0,0],
				[0,5,3,0,0,0]]
	
	kingdomNames1 = ['a' + str(i) for i in range(1, 6)]
	kingdomNames2 = ['a' + str(i) for i in range(1, 7)]
	
	findMinTour(0, fullWalk_1, cqList_1, matrix_1, len(matrix_1), kingdomNames1) == ['a1', 'a4']
	findMinTour(0, fullWalk_1, cqList1_1, matrix_1, len(matrix_1), kingdomNames1) == ['a1', 'a2', 'a1', 'a4']
	findMinTour(0, fullWalk_2, cqList_2, matrix_1, len(matrix_1), kingdomNames2) == ['a1', 'a2', 'a6', 'a2', 'a1', 'a4']


	"""

	# define a path 

	dijkstraPairs = {}
	g = Graph(maxtrix_size)
	if kingdomNames[startingKingdomIndex] not in cqList:
		cqList = cqList + [kingdomNames[startingKingdomIndex]]
	allpossiblePairs = list(itertools.combinations(cqList,2))
	for x in allpossiblePairs:
		dijkstraPairs[x] = g.dijkstra(matrix, kingdomNames, kingdomNames.index(x[0]), kingdomNames.index(x[1]))
	newGraphDict = {}
	newMatrix = [['x' for column in range(len(cqList))] for row in range(len(cqList))]
	startingIndex = 0
	for i in range(len(cqList)):
		newGraphDict[i] = cqList[i]
		if cqList[i] == kingdomNames[startingKingdomIndex]:
			startingIndex = i
	for i in range(len(cqList)):
		for j in range(len(cqList)):
			if (newGraphDict[i],newGraphDict[j]) in allpossiblePairs:
				newMatrix[i][j] = travellingcost(matrix, dijkstraPairs[(newGraphDict[i], newGraphDict[j])], kingdomNames)
			elif (newGraphDict[j], newGraphDict[i]) in allpossiblePairs:
				newMatrix[i][j] = travellingcost(matrix, dijkstraPairs[(newGraphDict[j], newGraphDict[i])], kingdomNames)
	list_of_kingdoms_to_visit = set(newGraphDict.keys())
	list_of_kingdoms_to_visit.remove(startingIndex)
	path = TSP(newMatrix, list_of_kingdoms_to_visit, startingIndex, startingIndex)[0]
	original_path = []
	for x in path:
		original_path.append(newGraphDict[x])
	if (original_path[0] != kingdomNames[startingKingdomIndex]):
		original_path = [kingdomNames[startingKingdomIndex]] + original_path
	returned_path = []
	for i in range(len(original_path) - 1):
		if (original_path[i], original_path[i+1]) in allpossiblePairs:
			returned_path = returned_path + [original_path[i]] + dijkstraPairs[(original_path[i], original_path[i+1])][:-1]
		elif (original_path[i + 1], original_path[i]) in allpossiblePairs:
			returned_path = returned_path + list(reversed(dijkstraPairs[(original_path[i + 1], original_path[i])]))
	returned_path = returned_path + [kingdomNames[startingKingdomIndex]]
	return returned_path

# Python program for Dijkstra's single source shortest
# path algorithm. The program is for adjacency matrix
# representation of the graph
 
from collections import defaultdict
 
#Class to represent a graph
class Graph:
	def __init__(self,vertices):
		self.V = vertices
		self.graph = [[0 for column in range(vertices)] for row in range(vertices)]
	def minDistance(self,dist,queue):
		minimum = float("inf")
		min_index = -1
		for i in range(len(dist)):
			if dist[i] < minimum and i in queue:
				minimum = dist[i]
				min_index = i
		return min_index

	def printPath(self,parent,j):
		path = []
		if parent[j] == -1:
			path.append(j)
			#print(j)
			return
		self.printPath(parent, parent[j])
		path.append(j)
		#print(j)
		#print(path)
		return path
	def printSolution(self,dist,parent,dst):
		src = 0
		print("Vertex Distance from Source Path")
		lst_paths = {}
		for i in range(1, len(dist)):
			print(src,i, dist[i])
			self.printPath(parent,i)
		#print(lst_paths[dst])
		#return lst_paths[dst]
	def returnPath(self,parent,j):
		path = []
		#print(parent)
		if parent[j] == -1:
			#path += [j]
			return path
		path.append(j)
		return self.returnPath(parent,parent[j])
		

	def dijkstra(self, graph,kingdomNames,src,dst):
		row = len(graph)
		col = len(graph[0])
		dist = [float("Inf")] * row
		parent = [-1] * row
		dist[src] = 0
		queue = []
		paths = []
		#print("Starting kingdom: ", src)
		#print("Ending kingdom: ", dst)
		for i in range(row):
			queue.append(i)
		#Find shortest path for all vertice
		while queue:
		# Pick the minimum dist vertex from the set of vertices
		# still in queue
			u = self.minDistance(dist,queue)
		# remove min element     
			queue.remove(u)
			for i in range(col):
				if graph[u][i] != 'x' and i in queue:
					if dist[u] + graph[u][i] < dist[i]:
						dist[i] = dist[u] + graph[u][i]
						parent[i] = u
		# print the constructed distance array
		#paths = self.printSolution(dist,parent,dst)
		#print(path)
		#return path

		#return self.returnPath(src,dst)
		#print(parent[dst])
		#print(dst)
		while parent[dst] != -1:
			paths.append(kingdomNames[dst])
			dst = parent[dst]

		paths.reverse()
		return paths

def unsurrenededNeighbors(matrix, conquered, conqueredList):
	neigh = []
	for i in range(0, len(matrix)):
		if i not in conquered:
			neighi = []
			for j in range(0, len(matrix[i])):
				if matrix[i][j] != 'x' and i != j and j not in conqueredList:
					neighi.append(j)
			if (neighi != []):
				neigh.append((neighi, i))
	return neigh

def maxEffectiveNeighbor(startingKingdom, kingdomNames, matrix, neighbors):
	g = Graph(len(matrix))
	if (len(neighbors) == 0):
		return ([],0)
	result = neighbors[0]
	for x in neighbors:
		if float(matrix[x[1]][x[1]]) / float(len(x[0])) < float(matrix[result[1]][result[1]])/ float(len(result[0])):
			result = x
	return result

def setCoverGreedy(startingKingdom, kingdomNames, matrix):
	result = []
	cqred = set()
	while (len(cqred) != len(matrix)):
		maxNeigh = maxEffectiveNeighbor(startingKingdom, kingdomNames, matrix, unsurrenededNeighbors(matrix, result, cqred))
		cqred.add(maxNeigh[1])
		cqred.update(maxNeigh[0])
		result.append(maxNeigh[1])
	resultnames = []
	for x in result:
		resultnames.append(kingdomNames[x])
	return resultnames



def solve(list_of_kingdom_names, starting_kingdom, adjacency_matrix, params=[]):
	"""
    Write your algorithm here.
    Input:
        list_of_kingdom_names: An list of kingdom names such that node i of the graph corresponds to name index i in the list
        starting_kingdom: The name of the starting kingdom for the walk
        adjacency_matrix: The adjacency matrix from the input file

    Output:
        Return 2 things. The first is a list of kingdoms representing the walk, and the second is the set of kingdoms that are conquered
    """
	#A = adjacency matrix, u = vertex u, v = vertex v
	def weight(A, u, v):
		return A[u][v]

	#A = adjacency matrix, u = vertex u
	def adjacent(A, u):
		L = []
		for x in range(len(A)):
			if  A[u][x] != 'x' and A[u][x] > 0 and x != u:
				L.insert(0,x)
		return L

	#Q = min queue
	def extractMin(Q):
		q = Q[0]
		Q.remove(Q[0])
		return q

		#Q = min queue, V = vertex list
	def decreaseKey(Q, K):
		for i in range(len(Q)):
			for j in range(len(Q)):
				if K[Q[i]] < K[Q[j]]:
					s = Q[i]
					Q[i] = Q[j]
					Q[j] = s

	#V = vertex list, A = adjacency list, r = root
	def prim(V, A, r):

		u = 0
		v = 0
		# initialize and set each value of the array P (pi) to none
		# pi holds the parent of u, so P(v)=u means u is the parent of v
		P=[None]*len(V)

		# initialize and set each value of the array K (key) to some large number (simulate infinity)
		K = [float('inf')]*len(V)

		# initialize the min queue and fill it with all vertices in V
		Q=[0]*len(V)
		for u in range(len(Q)):
			Q[u] = V[u]
		# set the key of the root to 0
		K[r] = 0
		# print(K)
		decreaseKey(Q, K)    # maintain the min queue
		# loop while the min queue is not empty
		while len(Q) > 0:
			u = extractMin(Q)    # pop the first vertex off the min queue
			# loop through the vertices adjacent to u
			Adj = adjacent(A, u)
			for v in Adj:
				w = weight(A, u, v)    # get the weight of the edge uv

				# proceed if v is in Q and the weight of uv is less than v's key
				if Q.count(v)>0 and w < K[v]:
				# set v's parent to u
					P[v] = u
					# v's key to the weight of uv
					K[v] = w
					decreaseKey(Q, K)    # maintain the min queue
		return P

	names = [x for x in range(len(list_of_kingdom_names))]
	graph = prim(names, adjacency_matrix, list_of_kingdom_names.index(starting_kingdom))

	# key = parent, value = children
	g = {}

	for x in range(len(list_of_kingdom_names)):
		g[x] = []

	for x in range(len(graph)):
		for y in range(len(graph)):
			if x == graph[y]:
				g[x].append(y)  

	def path(k):
		if not g[k]:
			return [k]

		lst = [k]

		for child in g[k]:
			lst += path(child) + [k]
			# print(lst)

		return lst

	def index_to_name(path):
		return [list_of_kingdom_names[x] for x in path]
	sol = []
	if (len(list_of_kingdom_names) <= 20) :
		sol = solver0.solve(list_of_kingdom_names, starting_kingdom, adjacency_matrix)


	conqueredKingdoms = setCoverGreedy(list_of_kingdom_names.index(starting_kingdom), list_of_kingdom_names, adjacency_matrix)
	if (len(conqueredKingdoms) <= 8):
		closed_walk = findMinTourAlternative(list_of_kingdom_names.index(starting_kingdom), conqueredKingdoms, adjacency_matrix, len(adjacency_matrix), list_of_kingdom_names)
	else:
		full_path = path(list_of_kingdom_names.index(starting_kingdom))
		full_path = index_to_name(full_path)
		closed_walk = findMinTour(list_of_kingdom_names.index(starting_kingdom), full_path, conqueredKingdoms, adjacency_matrix, len(adjacency_matrix), list_of_kingdom_names)
	closed_walk2 = findMinTourAlternative3(list_of_kingdom_names.index(starting_kingdom), conqueredKingdoms, adjacency_matrix, len(adjacency_matrix), list_of_kingdom_names)
	if (totalcost(adjacency_matrix, conqueredKingdoms, closed_walk2, list_of_kingdom_names) < totalcost(adjacency_matrix, conqueredKingdoms, closed_walk, list_of_kingdom_names)):
		closed_walk = closed_walk2
	if sol != []:
		if totalcost(adjacency_matrix, sol[1], sol[0], list_of_kingdom_names) < totalcost(adjacency_matrix, conqueredKingdoms, closed_walk, list_of_kingdom_names):
			closed_walk = sol[0]
			conqueredKingdoms = sol[1]
	return closed_walk, conqueredKingdoms




def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)
    
    input_data = utils.read_file(input_file)
    number_of_kingdoms, list_of_kingdom_names, starting_kingdom, adjacency_matrix = data_parser(input_data)
    closed_walk, conquered_kingdoms = solve(list_of_kingdom_names, starting_kingdom, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    output_filename = utils.input_to_output(filename)
    output_file = f'{output_directory}/{output_filename}'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    utils.write_data_to_file(output_file, closed_walk, ' ')
    utils.write_to_file(output_file, '\n', append=True)
    utils.write_data_to_file(output_file, conquered_kingdoms, ' ', append=True)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    input_directory = args.input
    solve_all(input_directory, output_directory, params=args.params)


