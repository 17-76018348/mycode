#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int findlen(int num);
int* createStack(int n);
void countStack(int* stack, int dec, int bin_len);
int main(void)
{
    int dec, bin;
    int bin_len;
    int* stack;

    printf("숫자를 입력하세요\n");
    scanf("%d",&dec);
    bin_len = findlen(dec);
    stack = createStack(bin_len);
    countStack(stack, dec, bin_len);
    
    int i = 0;
    for(i = 0;i<bin_len; i++)
    {
        printf("%d\t",stack[i]);
    }

    
    return 0;
}
int findlen(int num)
{
    float dec = (float)num;
    int cnt = 0;
    int idx = 0;
    while(idx < 100)
    {
        if(dec >= 2)
        {
            dec /= 2;
            cnt += 1;
        }
        else
        {
            return ++cnt;
        }    
        idx += 1;
    }
}
int* createStack(int n)
{
    int *stack;
    stack = malloc(sizeof(int) * n);
    printf("스택이 생성되었습니다\n");
    return stack;
}
void countStack(int* stack, int dec, int bin_len)
{
    int idx;
    int cnt = 0;
    for(idx = bin_len - 1; idx >= 0; idx--)
    {
        if(dec >= pow(2,idx))
        {
            dec -= pow(2,idx);
            stack[cnt] = 1;
        }
        else
        {
            stack[cnt] = 0;
        }
        cnt += 1;
    }
}