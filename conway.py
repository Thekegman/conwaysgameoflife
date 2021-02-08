import numpy as np
from scipy.signal import convolve
from timeit import Timer
import matplotlib.animation as animation
from matplotlib import pyplot as plt

conway_kernal = np.ones([3,3],np.int8)
conway_kernal[1,1] = 0

def blank_board(x, y):
    return np.zeros([y,x],np.int8)
def random_board(x, y):
    return np.array(np.random.randint(2, size=(y,x)),np.int8)
def next_state_conv(board):
    neighbour_count= np.round(convolve(board, conway_kernal)[1:-1,1:-1])
    board[neighbour_count<2]  = 0
    board[neighbour_count>3]  = 0
    board[neighbour_count==3] = 1


def next_state_simple(board):
    neighbour_count = np.zeros_like(board)
    # add top
    neighbour_count[1:,:] += board[:-1,:]
    # add bottom
    neighbour_count[:-1,:] += board[1:,:]
    # add right
    neighbour_count[:,:-1] += board[:,1:]
    # add left
    neighbour_count[:,1:] += board[:,:-1]
    # add top left
    neighbour_count[1:,1:] += board[:-1,:-1]
    # add top right
    neighbour_count[1:,:-1] += board[:-1,1:]
    # add bottom left
    neighbour_count[:-1,1:] += board[1:,:-1]
    # add bottom right
    neighbour_count[:-1,:-1] += board[1:,1:]
    
    board[neighbour_count<2]  = 0
    board[neighbour_count>3]  = 0
    board[neighbour_count==3] = 1

def add_glider(board,x,y):
    board[y,x-1:x+2]=1
    board[y-1,x+1]=1
    board[y-2,x]=1



def updatefig(*args):
    next_state_simple(board)
    im.set_array(view)
    return im,

def animate():
    global im
    fig = plt.figure()
    im = plt.imshow(view, animated=True)
    ani = animation.FuncAnimation(fig, updatefig, interval=17,  blit=True)
    plt.show()

board = blank_board(404, 404)
view = board[2:-2,2:-2]

glider_count = 20
for x,y in np.random.randint(view.shape, size=(glider_count,2)):
    add_glider(board, x,y)


"""
t1 = Timer(lambda :next_state_simple(board))
t2 = Timer(lambda :next_state_conv(board))

print(t1.timeit(number=100))
print(t2.timeit(number=100))
"""
animate()
