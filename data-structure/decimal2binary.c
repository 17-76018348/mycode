#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int count(int num, int* stack, int* chk);
int* createStack(int n);
void push(int target, int* stack, int* chk);
int pop(int* stack, int* chk);
int isEmptyStack(int* stack, int* chk);
int main(void)
{
    int* stack;
    int* chk = malloc(sizeof(int));
    chk[0] = -1;
    int n = 10;
    int dec;
    dec = 10;
    stack = createStack(n);
    int tmp = 0;
    int tmp2 = -1;
    tmp = count(dec, stack, chk);
    // printf("%d",chk[0]);
    // tmp2 = pop(stack,chk);
    // while(1)
    // {
    //     if(tmp2 == 0)
    //     {
    //         return 0;
    //     }
    //     tmp2 = pop(stack,chk);
    // }



    return 0;
}
int count(int num, int* stack, int* chk)
{
    printf("%d\n",num);
    int result;
    int out;
    if(result == 0)
    {
        return 1;
    }

    result = num / 2;
    out = num % 2;
    push(out, stack, chk);
    return count(result, stack, chk);
    
}
int* createStack(int n)
{
    int *stack;
    stack = malloc(sizeof(int) * n);
    printf("스택이 생성되었습니다\n");
    return stack;
}
void push(int target, int* stack, int* chk)
{
    //isfullstack
    stack[chk[0] + 1] = target;
    printf("%d\n",stack[chk[0] + 1]);
    chk[0] += 1;
}
int pop(int* stack, int* chk)
{
    if(isEmptyStack(stack, chk) == 1)
    {
        printf("끝");
        return 0;
    }
    int data;
    printf("%d\t",stack[chk[0]]);
    data = stack[chk[0]];
    chk[0] -= 1;
    return 1;
}
int isEmptyStack(int* stack, int* chk)
{
    if(chk[0] == -1)
    {
        return 1;
    }
    else
    {
        return 0;
    }
    
}
