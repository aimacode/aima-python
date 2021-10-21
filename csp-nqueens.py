import numpy as np


# 3 constraints 
# no queens on the same row 
# no queens on the same col 
# no queens on the same diagonal


def is_safe_row(board,n):
    for row in range(n):
        if sum(board[row,:]) > 1: # if there is a row with more than one queen, then its sum will be more than 1s
            return False
    return True
        
def is_safe_col(board,n):
    for col in range(n):
        if sum(board[:,col]) > 1 : 
            return False
    return True

def is_safe_diag(board,n):
    diags = []
    for i in range(-n+1,n):
        # if the index i is negative, take the upper ith diagonal
        # if the index i is positive, the the lower ith diagonal
        diags.append(board.diagonal(i))
                    
    diags.extend(board.diagonal(i) for i in range(n-1,-n,-1))
    for x in diags:
        if len(x)>1:
            if sum (x)>1:
                return False
    return True


def nqueens (board, queen, n):
    if is_safe_row(board,n) and is_safe_col(board,n) and is_safe_diag(board,n):
        if board.sum()==n:
            return True

        for row in range (0,n):
            board[row,queen] = 1
            print(f'Queen: {row}')
            if is_safe_row(board,n) and is_safe_col(board,n) and is_safe_diag(board,n):
                if nqueens(board, queen+1,n):
                    return True
                board[row,queen] = 0
            else:
                board[row,queen] = 0
    return False


if __name__ == '__main__':
    n = 4
    board = np.zeros((n, n))
    if nqueens(board, 0, n):
        print(board)
    else:
        print('Cannot find a solution for ',n,' queens.')