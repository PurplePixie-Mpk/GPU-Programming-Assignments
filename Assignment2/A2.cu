#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

__global__ void sumRandC(int* A, int* B, int m, int n, int k=1)
{
    // m -> number of rows in A
    // n-> number of columns in A
    int id;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int _id = threadId * k;
    for(id = _id; id<(k+_id); ++id)
    {
        int cno,rno;
        cno = id%n;
        rno = id/n;
        // B has capital indexing
        int N = n+1;
        // int M = m+1;
        // matrix dimensions m,n:
        // A[rno][cno]-> id = rno*n+cno;
        if(rno<m)
        {
            // What we are actually doing (in sequential):
            // B[rno][cno]=A[rno][cno]; //A[rno][cno] = A[id];
            // B[rno][n] += A[rno][cno];
            // B[m][cno] += A[rno][cno];
            int idB = rno*N + cno;
            B[idB] = A[id];
            atomicAdd(&B[rno*N + n], A[id]);
            atomicAdd(&B[m*N + cno], A[id]); 
            atomicAdd(&B[m*N + n], A[id]); //Adding all elements to bottom right corner
        }
    }
}

__global__ void findMin(int* B, int m, int n, int k=1)
{
    int id;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int _id = threadId * k;
    for(id = _id; id<(k+_id); ++id)
    {
        int cno,rno;
        cno = id%(n+1);
        rno = id/(n+1);
        if(rno<=m)
        {
            if(cno == n)
            {
                atomicMin(&B[m*(n+1) + n], B[rno*(n+1) + cno]);
            }
            else if(rno == m)
            {
                atomicMin(&B[m*(n+1) + n], B[rno*(n+1) + cno]);
            }
        }
    }
}

__global__ void updameMin(int* A, int* B, int m, int n, int k=1)
{
    int id;
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int _id = threadId * k;
    int mn = B[m*(n+1) + n];
    for(id = _id; id<(k+_id); ++id)
    {
        int cno,rno;
        cno = id%(n+1);
        rno = id/(n+1);
        if(rno<m && cno<n)
        {
            B[id] += mn;
        }
    }
}

int main()
{
    int n,i,j,m,k;
    scanf("%d", &m);
    scanf("%d", &n);
    scanf("%d", &k);
    int *a,*b,*sol, *A;
    a = (int *) malloc (m*n*sizeof(int));
    sol = (int *) malloc ((m+1)*(n+1)*sizeof(int));
    // Reading in input
    for(i=0;i<m*n;++i)
        scanf("%d", &a[i]);

    cudaMalloc(&b,(n+1)*(m+1)*sizeof(int));
    cudaMalloc(&A,n*m*sizeof(int));
    cudaMemcpy(A,a,n*m*sizeof(int),cudaMemcpyHostToDevice);

    // Block dimensions can be changed here:
    dim3 blockD(5,5,1);
    // Grid dimensions can be changed here:
    dim3 gridD(5,5,1);
    
    // k for each function is the last argument inthe following function calls and can be changed:
    sumRandC<<<gridD,blockD>>>(A,b,m,n,2);
    findMin<<<gridD,blockD>>>(b,m,n,3);
    updameMin<<<gridD,blockD>>>(A,b,m,n,4);
    
    cudaDeviceSynchronize();
    cudaMemcpy(sol,b,(n+1)*(m+1)*sizeof(int),cudaMemcpyDeviceToHost);
    for(i=0;i<(m+1);i++)
    {
        for(j=0;j<(n+1);j++)
            printf("%d ",sol[(n+1)*i+j]);
        printf("\n");
    }
    return 0;
}
