#include <stdio.h>
int* createStack(int n);
void countStack(int* stack);
void push(int target, int* stack, int* chk);
void pop(int* stack, int* chk);
int main(void)
{
    int dec;
    int* stack;
    printf("decimal 입력하세요\n");
    scanf("%d",&dec);
    stack = createStack(dec);

    return 0;
}
int* createStack(int n)
{
    int *stack;
    stack = malloc(sizeof(int) * n);
    printf("스택이 생성되었습니다\n");
    return stack;
}
void countStack(int* stack)
{

}
void push(int target, int* stack, int* chk)
{
    //isfullstack
    stack[chk[0] + 1] = target;
    printf("%d이 삽입되었습니다\n",stack[chk[0] + 1]);
    chk[0] += 1;
}
void pop(int* stack, int* chk)
{
    //isemptystack
    int data;
    printf("%d이 제거되었습니다\n",stack[chk[0]]);
    data = stack[chk[0]];
    chk[0] -= 1;
}