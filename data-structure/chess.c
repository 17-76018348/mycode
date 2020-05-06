#include <stdio.h>
#include <stdlib.h>
void posit(int lev, int** array);
int** initial();
void reset(int** array);
int main()
{
	int** array;
    array = initial();
	posit(0, array);

    return 0;
}
void posit(int lev, int** array)
{

    int x;
    int i;
    
    for(x = 0;x < 8; x++)
    {
        if(array[0][x] == 0 && array[1][lev + x] == 0 && array[2][lev - x + 7] == 0)
        {
            array[3][lev] = x;
            if(lev == 7)
            {
            	for(i = 0;i < 8; i++)
            	{
            		printf("(%d,%d)\t",i,array[3][i]);
				}
				printf("\n");
			}
			else
			{
			
				array[0][x] = 1;
				array[1][lev + x] = 1;
				array[2][lev - x + 7] = 1;
				posit(lev + 1,array);
				array[0][x] = 0;
				array[1][lev + x] = 0;
				array[2][lev - x + 7] = 0;
			}

        }
    }
}
int** initial()
{
    int *vertic;
    vertic = (int*)malloc(sizeof(int) * 8);
    int i;
    for(i = 0;i < 8; i++)
    {
        vertic[i] = 0;
    }
    int *cross_1;
    cross_1 = (int*)malloc(sizeof(int) * 15);
    for(i = 0;i < 15; i++)
    {
        cross_1[i] = 0;
    }
    int *cross_2;
    cross_2 = (int*)malloc(sizeof(int) * 15);
    for(i = 0;i < 15; i++)
    {
        cross_2[i] = 0;
    }
    int *sequence;
    sequence = (int*)malloc(sizeof(int) * 8);
    for(i = 0;i < 8; i++)
    {
        sequence[i] = 0;
    }
    int** array;
    array = (int*)malloc(sizeof(int *) * 4);
    array[0] = vertic;
    array[1] = cross_1;
    array[2] = cross_2;
    array[3] = sequence;
    return array;
}
void reset(int** array)
{
    int i;
    for(i = 0;i < 8; i++)
    {
        array[0][i] = 0;
        array[3][i] = 0;
    }
    for(i = 0;i < 15; i++)
    {
        array[1][i] = 0;
        array[2][i] = 0;
    }
    
}
