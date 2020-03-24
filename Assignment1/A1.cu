__global__ void per_row_kernel(int *in, int N) //1D grid, 2D block
{
    int cno;
    int id = threadIdx.x*blockDim.y + threadIdx.y + blockIdx.x*(blockDim.x * blockDim.y);
    int rno = id;
    if(rno<N)
    {
        for(cno=0;cno<N;++cno) //i is column number
        {
            if(rno>cno)
            {
                in[cno*N + rno] = in[rno*N + cno];
                in[rno*N + cno] = 0;
            }
        }
    }
}

__global__ void per_element_kernel(int *in, int N) //3D grid, 1D block
{
    int id = blockIdx.z*blockDim.x*gridDim.x*gridDim.y + blockIdx.y*blockDim.x*gridDim.x + blockIdx.x*blockDim.x + threadIdx.x;
    int cno,rno;
    cno = id%N;
    rno = id/N;
    if(rno<N)
    {
        if(rno > cno)
        {
            in[cno*N + rno] = in[rno*N + cno];
            in[rno*N + cno] = 0;
        }
    }    
}

__global__ void per_element_kernel_2D(int *in, int N) //2D grid, 2D block
{
    int id = blockIdx.y*blockDim.x*blockDim.y*gridDim.x + blockIdx.x*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x;
    int cno,rno;
    cno = id%N;
    rno = id/N;
    if(rno<N)
    {
        if(rno > cno)
        {
            in[cno*N + rno] = in[rno*N + cno];
            in[rno*N + cno] = 0;
        }
    }
}
