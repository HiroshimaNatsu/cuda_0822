#!/usr/bin/env python3
import numpy 
import curses
from curses import wrapper
import time
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
from pycuda.compiler import SourceModule

BLOCKSIZE = 32
GPU_NITER = 1

row2str = lambda row:''.join('0' if c != 0 else ' ' for c in row)
call_value = lambda world, height, width,y,x:world[y % height, x % width]


mod = SourceModule("""
__global__ void add_matrix_gpu(const int* __restrict__ dMat_A,int *dMat_G,const int mat_size_x,const int mat_size_y ){
    int mat_x = threadIdx.x + blockIdx.x * blockDim.x;
    int mat_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (mat_x >= mat_size_x) {
        return;
    }
    if (mat_y >= mat_size_y) {
        return;
    }
    const int index = mat_y * mat_size_x + mat_x;
    int current_value = dMat_A[index];
    int next_value = current_value;
    int num_live = 0;

    num_live += dMat_A[((mat_y-1)%mat_size_y)*(mat_size_x)  +  ((mat_x-1)%mat_size_x)];
    num_live += dMat_A[((mat_y-1)%mat_size_y)*(mat_size_x)  +  ((mat_x)%mat_size_x)];
    num_live += dMat_A[((mat_y-1)%mat_size_y)*(mat_size_x)  +  ((mat_x+1)%mat_size_x)];
    num_live += dMat_A[((mat_y)%mat_size_y)*(mat_size_x)  +  ((mat_x-1)%mat_size_x)];
    num_live += dMat_A[((mat_y)%mat_size_y)*(mat_size_x)  +  ((mat_x+1)%mat_size_x)];
    num_live += dMat_A[((mat_y+1)%mat_size_y)*(mat_size_x)  +  ((mat_x-1)%mat_size_x)];
    num_live += dMat_A[((mat_y+1)%mat_size_y)*(mat_size_x)  +  ((mat_x)%mat_size_x)];
    num_live += dMat_A[((mat_y+1)%mat_size_y)*(mat_size_x)  +  ((mat_x+1)%mat_size_x)];
    

    if( current_value == 0 && num_live == 3){
        next_value = 1;
    }else if(current_value == 1 && (num_live == 2 || num_live == 3)){
        next_value = 1;
    }else{
        next_value = 0;
    }
    dMat_G[index] = next_value;

}
""")
def set_next_call_value_gpu(world,next_world,height,width):
    add_matrix_gpu = mod.get_function("add_matrix_gpu")
    block = (BLOCKSIZE,BLOCKSIZE,1)
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])
    #print("Grid = ({0},{1}),Block=({2},{3})".format(grid[0],grid[1],block[0],block[1]))
    start = cuda.Event()
    end = cuda.Event()
    start.record()
    for i in range(GPU_NITER):
        add_matrix_gpu(cuda.In(world),cuda.Out(next_world),numpy.int32(width),numpy.int32(height),block=block,grid=grid)
    end.record()
    #end.synchronize()
    #elapsed_sec = start.time_till(end) * 1e-3 / GPU_NITER

def print_world(stdscr,world,generation,elapsed):
    height,width = world.shape
    for y in range(height):
        row = world[y]
        stdscr.addstr(y,0,row2str(row))
    stdscr.addstr(height,0,"Generation:%06d,Elapsed:%.6f[sec]"%(generation,elapsed/generation))
    stdscr.refresh()


def set_next_call_value(world,next_world,height,width,y,x):
    current_value = call_value(world,height,width,y,x)
    next_value = current_value
    num_live = 0
    num_live += call_value(world,height,width,y-1,x-1)
    num_live += call_value(world,height,width,y-1,x)
    num_live += call_value(world,height,width,y-1,x+1)
    num_live += call_value(world,height,width,y,x-1)
    num_live += call_value(world,height,width,y,x+1)
    num_live += call_value(world,height,width,y+1,x-1)
    num_live += call_value(world,height,width,y+1,x)
    num_live += call_value(world,height,width,y+1,x+1)
    if current_value == 0 and num_live == 3:
        next_value = 1
    elif current_value == 1 and num_live in(2,3):
        next_value = 1
    else:
        next_value = 0
    next_world[y,x] = next_value

def calc_next_world_gpu(world,next_world):
    height,width = world.shape
    set_next_call_value_gpu(world,next_world,height,width)


def calc_next_world_cpu(world,next_world):
    height,width = world.shape
    for y in range(height):
        for x in range(width):
            set_next_call_value(world,next_world,height,width,y,x)


def game_of_life(stdscr,height,width):
    #世界の初期値
    world = numpy.random.randint(2,size=(height,width),dtype=numpy.int32)
    
    #次の世代を格納する２次元配列
    next_world = numpy.empty((height,width), dtype=numpy.int32)

    elapsed = 0.0
    generation = 0
    while True:
        generation += 1
        print_world(stdscr,world,generation,elapsed)
        start_time = time.time()
        #calc_next_world_cpu(world,next_world)
        calc_next_world_gpu(world,next_world)
        duration = time.time() - start_time
        elapsed += duration
        world,next_world = next_world,world

def main(stdscr):
    stdscr.clear()
    stdscr.nodelay(True)
    scr_height , scr_width = stdscr.getmaxyx()
    game_of_life(stdscr,scr_height - 1,scr_width)



if __name__ == '__main__':
    curses.wrapper(main)