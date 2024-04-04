import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
#csr compressed sparse row - condenses 0 values to index location reference making the matrix storage smaller
# creating a sparse matrix from a csv
dataframe = pd.read_csv("simd2020_withinds_0s.csv")
print(dataframe.head(2))
#drop non-numeric data
dataframe = dataframe.select_dtypes(include=[np.number])
# Drop empty rows
dataframe = dataframe.dropna(axis=0, how='all')
# Drop empty columns
dataframe= dataframe.dropna(axis=1, how='all')
dataframe.to_csv("numeric_only.csv")
#dataframe_numeric = pd.read_csv("numeric_only.csv", header=0)

print(dataframe.head(2))

print("dataframe shape before csr:", dataframe.shape)
#matrix_sparse = sparse.csr_matrix(matrix)
sparse_matrix=csr_matrix(dataframe.values)
print("dataframe shape after csr:", sparse_matrix.shape)

# size and shape are the same
#to check if it's worked I need to check memory usage - sparse should be smaller, but on a data size this size it will
#be a small difference
# Calculate memory usage for the dense DataFrame
dense_memory_usage = dataframe.memory_usage(index=True).sum()



# Calculate memory usage for the sparse matrix
# This includes the size of data, indices, and indptr arrays
sparse_memory_usage = (
    sparse_matrix.data.nbytes + 
    sparse_matrix.indices.nbytes + 
    sparse_matrix.indptr.nbytes
)

print("Dense DataFrame memory usage (bytes):", dense_memory_usage)
print("Sparse matrix memory usage (bytes):", sparse_memory_usage)

# this data set is not sparse enough to show a smaller use, the index refrences potentially take up more space
#but the sytax and methodology for doing it seems correct.