import sys
import numpy as np
from pyspark import SparkConf,SparkContext

def output(W, W_out_path, H, H_out_path):
	np.savetxt(W_out_path, W, delimiter = ',')
	np.savetxt(H_out_path, H, delimiter = ',')
	return

#this function performs gradient decent in each strata
#returns a list of tuples of modified parameters.
#in each tuple can find the index and updated value of W and H vector
def gradient_decent(x, W, H, i, parameters, mbi):
	beta_value, lambda_val = parameters.value
	n_ = 0
	updated = []
	w_set, h_set = set(), set()
	for entry in x:
		epsilon = (100 + mbi + n_) ** -beta_value
		i, j, rating, n_i, n_j = entry[1]
		ratio = -2 * (rating - np.dot(W[i], H[j]))
		W[i] -= epsilon * (ratio * H[j] + 2 * lambda_val / n_i*W[i])
		H[j] -= epsilon * (ratio * W[i] + 2 * lambda_val / n_j*H[j])
		w_set.add(i)
		h_set.add(j)
		n_ += 1
	for i in w_set:
		updated.append(("W", i, W[i]))
	for j in h_set:
		updated.append(("H", j, H[j]))
	return updated

#this fuction add the first key, which is used to partition data into different stratas
def add_first_key(x, row_block_len, col_block_len, num_workers):
	i_block = int(x[1] / row_block_len)
	j_block = int(x[2] / col_block_len)
	key = (j_block + num_workers - i_block) % num_workers
	return (key, x[0], x[1], x[2], x[3], x[4], x[5])

#this function add the secondary key, which is used to partion strata into different workers
def add_secondary_key(x, row_block_len, n_is, n_js):
	return (int(x[0]/row_block_len), x[0], x[1], x[2], n_is[x[0]], n_js[x[1]])
	
def main(argv):
	#initialize each of the parameters
	num_factors = int(argv[1])
	num_workers = int(argv[2])
	num_iterats = int(argv[3])
	beta_value = float(argv[4])
	lambda_val = float(argv[5])
	V_path = argv[6]
	W_out_path = argv[7]
	H_out_path = argv[8]

	#setup
	sc = SparkContext(appName = "pigwall") 

	#read in as sparse matrix tuple
	disFile = sc.textFile(V_path)
	sparse_matrix = disFile.map(lambda x:x.split(',')[0:3]).cache()
	sparse_matrix = sparse_matrix.map(lambda x:[int(x[0])-1,int(x[1])-1,int(x[2])])

	#get max user id and max movie id
	max_row_id = sparse_matrix.max(lambda x:x[0])[0]
	max_col_id = sparse_matrix.max(lambda x:x[1])[1]

	#get block size, used in partition
	row_block_len = int((max_row_id+1)/num_workers)
	col_block_len = int((max_col_id+1)/num_workers)

	#get n_i*,n_j*
	n_is = sparse_matrix.map(lambda x:(x[0],1)).countByKey()
	n_js = sparse_matrix.map(lambda x:(x[1],1)).countByKey()

	#add two outlayer keys
	#first key is used to separate data into different stratas
	#sencond key is used to separate each strata into different workers
	new_matrix = sparse_matrix.map(lambda x:add_secondary_key(x,row_block_len,n_is,n_js))
	new_matrix = new_matrix.map(lambda x:add_first_key(x,row_block_len,col_block_len,num_workers)).cache()

	#create stratas
	stratas = []
	mbi_count = 0
	for i in range(0,num_workers):
		strata = new_matrix.filter(lambda x:x[0]==i).map(lambda x:(x[1], (x[2], x[3], x[4], x[5], x[6]))).partitionBy(num_workers).cache()
		stratas.append(strata)
		mbi_count += strata.count()

	#initiallze W,H, and broadcast beta and lambda
	W = np.random.rand(max_row_id+1,num_factors)
	H = np.random.rand(max_col_id+1,num_factors)
	parameters = sc.broadcast([beta_value, lambda_val])

	#iterations of GD
	mbi = 0
	for i in range(0,num_iterats):
		curriter_mbi = 0
		for strata in stratas:
			updates = strata.mapPartitions(lambda x:gradient_decent(x, W, H, i, parameters, mbi)).collect()

			for u in updates:
				if u[0] == 'W':
					W[u[1]] = u[2]
				elif u[0] == 'H':
					H[u[1]] = u[2]
		mbi += mbi_count
		#mse, n = sparse_matrix.map(lambda x:eval(x, W, H)).reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
		#print 'avg mse:', mse / n

	#output
	output(W, W_out_path, H.T, H_out_path)
	return

#def eval(x, W, H):
#	i, j, rating = x
#	return ((rating - np.dot(W[i], H[j]))**2, 1)

if __name__ == "__main__":
	main(sys.argv)