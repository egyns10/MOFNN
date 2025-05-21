#getData.py

def createGrid(headers):
    size = len(headers)
    grid = [['' for _ in range(size + 1)] for _ in range(size + 1)]
    for i in range(size):
        grid[0][i + 1] = headers[i]
        grid[i + 1][0] = headers[i]
    return grid

def intoArray(array,i,j,n):     
    #if arrayMSE[i+1,j+1] != "x":
    array[:,(i+1,j+1)] = n
    return array