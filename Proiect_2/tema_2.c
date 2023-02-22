#include "pgm_IO.h"
#include <mpi.h>
#include "pgm_IO.c"

int main(int argc, char **argv)
{
    int source, my_rank, nproc, tag = 1;
    int NX, NY, i, j, dims[NDIM], GSIZE = XSIZE * YSIZE;
    // NX: nr de procese distribuite pe directia Ox a structurii carteziene
    // NY: ... directia Oy ...
    // dims[NDIM]: contine distributia optimizata a proceselor pe structura carteziana
    // GSIZE: este dimensiunea blocului global care defineste imaginea

    int reorder = 0, periods[NDIM] = {TRUE, TRUE};
    // reorder: procesele din reteaua carteziana nu vor fi reordonate
    // structura carteziana 2D este periodica

    int dX, dY, localSize;
    // dX: XSIZE/NX
    // dY: XSIZE/NY
    // localSize = dX * dY este dimensiunea blocului de date stocat local pe fiecare procesor

    float *masterdata, *data;
    // masterdata: blocul XSIZE * YSIZE pixeli, stocat doar pe procesul 0
    // data: blocul local necesar fiecaruia dintr procese pentru a-si defini imaginea

    int coords[NDIM]; // sirul coordonatelor procesului in reteaua virtuala

    MPI_Comm wcomm = MPI_COMM_WORLD, comm_2D;
    // comm_2D: comunicatorul asociat retelei cu NDIM dimensiuni
    MPI_Status status;
    MPI_Datatype blockType; // tip de date derivat pentru transferul catre procesul 0 a datelor partiale

    char fname[32];
    strcpy(fname, "chessy_struct.pgm");

    /* Pas 1 */

    for(i = 0; i < NDIM; i++)
    {
        dims[i] = 0;
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Dims_create(nproc, NDIM, dims);
    NX = dims[0];
    NY = dims[1];

    MPI_Cart_create(wcomm, NDIM, dims, periods, reorder, &comm_2D);

    dX = XSIZE/NX;
    dY = YSIZE/NY;

    /* Pas 2 */

    if(my_rank == MASTER)
    {
        masterdata = (float*)malloc(XSIZE * YSIZE * sizeof(float));
        for(i = 0; i < XSIZE; i++)
        {
            for(j = 0; j < YSIZE; j++)
            {
                masterdata[i+j] = 0.75 * CONTRAST;
            }
        }
    }
    else
    {
        data = (float*)malloc((XSIZE/NX) * (YSIZE/NY) * sizeof(float));  
        for(i = 0; i < XSIZE/NX; i++)
        {
            for(j = 0; j < YSIZE/NX; j++)
            {
                data[i+j] = 0.75 * CONTRAST;
            }
        }
    }
    
    /* Pas 3 */

    MPI_Cart_coords(comm_2D, my_rank, NDIM, coords);
    if((coords[0] + coords[1] + 1) % 2 == 1)
    {
        for(i = 0; i < XSIZE/NX; i++)
        {
            for(j = 0; j < YSIZE/NX; j++)
            {
                data[i+j] = 0;
            }
        }
    }
    else
    {
        for(i = 0; i < XSIZE/NX; i++)
        {
            for(j = 0; j < YSIZE/NX; j++)
            {
                data[i+j] = CONTRAST;
            }
        }
    }
    MPI_Type_vector(dX, dY, YSIZE, MPI_FLOAT, &blockType);
    MPI_Type_commit(&blockType);

    MPI_Datatype lineType, colType;

    MPI_Type_vector(1, dY, 1, MPI_FLOAT, &lineType);
    MPI_Type_commit(&lineType);

    MPI_Type_vector(dX, 1, dY, MPI_FLOAT, &colType);
    MPI_Type_commit(&colType);

    int up, down, left, right;

    MPI_Cart_shift(comm_2D, 0, 1, &up, &down);
    MPI_Cart_shift(comm_2D, 1, 1, &left, &right);

    MPI_Sendrecv(data + dY, 1, lineType, up, 3, data + (dX - 1)*dY, 1, lineType, down, 3, comm_2D, &status);
    MPI_Sendrecv(data + (dY - 1), 1, lineType, down, 3, data + dX*dY, 1, lineType, up, 3, comm_2D, &status);

    MPI_Sendrecv(data + (dX - 1), 1, colType, right, 3, data + dX*dY, 1, colType, left, 3, comm_2D, &status);
    MPI_Sendrecv(data + (dX + 1)*dY, 1, colType, left, 3, data + (dX - 1)*dY, 1, colType, right, 3, comm_2D, &status);

    /* Pas 4 */

    MPI_Cart_coords(wcomm, my_rank, NDIM, coords);
    if(my_rank == MASTER)
    {
        for(i = 0; i < dX; i++)
        {
            for(j = 0; j < dY; j++)
            {
                masterdata[i * YSIZE + j] = data[i * dY + j];
            }
        }
        for(i = 1; i <= nproc - 1; i++)
        {
            MPI_Recv((masterdata + coords[1] * dY + coords[0] * dX * YSIZE), 1, blockType, i, tag, wcomm, &status);
        }
        pgm_write(fname, masterdata, NX, NY);
    }
    else
    {
        MPI_Send(data, dX * dY, MPI_FLOAT, MASTER, tag, wcomm);
    }

    free(masterdata);
    free(data);
    MPI_Finalize();

    return 0;
} //end of main()