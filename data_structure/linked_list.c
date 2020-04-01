#include <stdio.h>
int main(void)
{
    void *p = NULL;
    int i = 7;
    float f = 23.5;

    p = &i;
    p = &f;

    printf("i contiatins : %d\n", *((int*)p));
    printf("i contiatins : %f\n", *((float*)p));
    printf("$dddd");

    return 0;

}