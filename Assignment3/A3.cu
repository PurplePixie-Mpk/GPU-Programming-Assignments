#include <stdio.h>
#include <cuda.h>

__device__ int getnum(char *queryArray, int &i)
{
    int ans=0;
    while(queryArray[i]==' ' || queryArray[i]=='\t')
        i++;
    while(queryArray[i]<=57 && queryArray[i]>=48)
    {
        ans = ans*10 + (queryArray[i]-'0');
        i++;
    }
    while(queryArray[i]==' ' || queryArray[i]=='\t')
        i++;
    return ans;
}
__global__ void updateDB(int* dB, char* queryArray_GPU, int *queryIndices_GPU, int m, int n, int q)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < q)
    {
        int base = queryIndices_GPU[id];
        int i=base+3, j, rowno;
        int updval;
        int updcol;
        int colno = getnum(queryArray_GPU, i)-1;      
        int ncomp = getnum(queryArray_GPU, i); 
        int p = getnum(queryArray_GPU, i);
        queryIndices_GPU[id] = i;
        for(rowno = 0; rowno<m; rowno++)
        {
            if(dB[rowno*n + colno] == ncomp)
            {
                i = queryIndices_GPU[id];
                for(j = 0; j<p; j++)
                {
                    i+=1;
                    updcol = getnum(queryArray_GPU, i)-1;
                    updval = getnum(queryArray_GPU, i);
                    if(queryArray_GPU[i]=='-')
                        updval*=(-1);
                    i+=2;
                    atomicAdd(&dB[rowno*n + updcol], updval);
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
    FILE *inp, *otp;
    inp = fopen(argv[1], "r");
    int m, n, i, j;
    fscanf(inp, "%d", &m);
    fscanf(inp, "%d", &n);
    int *dataBase = (int *) malloc (m*n*sizeof(int));
    for(i=0;i<m*n;++i)
        fscanf(inp, "%d", &dataBase[i]);
    int *dB;
    cudaMalloc(&dB,n*m*sizeof(int));
    cudaMemcpy(dB,dataBase,n*m*sizeof(int),cudaMemcpyHostToDevice);
    int q;
    fscanf(inp, "%d", &q);
    int *queryIndices_CPU = (int *) malloc (q*sizeof(int));
    int *queryIndices_GPU;
    cudaMalloc(&queryIndices_GPU, q*sizeof(int));
    char *queryArray_CPU = (char *)malloc(q*300*sizeof(char));
    j=0;
    i=0;
    while(fscanf(inp, "%c", &queryArray_CPU[i]) != EOF)
    {
        if(queryArray_CPU[i] == 'U')
        {
            queryIndices_CPU[j] = i;
            j+=1;
        }
        i+=1;
    }
    char *queryArray_GPU;
    cudaMalloc(&queryArray_GPU, (i)*sizeof(char));
    fclose(inp);
    cudaMemcpy(queryIndices_GPU,queryIndices_CPU,q*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(queryArray_GPU,queryArray_CPU,(i)*sizeof(char),cudaMemcpyHostToDevice);
    updateDB<<<2, 1024>>>(dB, queryArray_GPU, queryIndices_GPU, m, n, q);
    cudaDeviceSynchronize();
    cudaMemcpy(dataBase,dB,n*m*sizeof(int),cudaMemcpyDeviceToHost);
    otp = fopen(argv[2], "w");
    for(i=0; i<m; ++i)
    {
        for(j=0; j<n; ++j)
            fprintf(otp, "%d ", dataBase[i*n + j]);
        fprintf(otp, "\n");
    } 
    return 0;
}
