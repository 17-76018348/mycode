#include <stdio.h>
#include <stdlib.h>
char* createStack(int n);
void push(char target, char* stack, int* chk);
void pop(char* stack, int* chk);
void stackPop(char *stack, int* chk);
void destroyStack(char *stack);
int main(void)
{
    char* stack;
    int *chk = malloc(sizeof(int));
    chk[0] = -1;
    char tmp_data;

    stack = createStack(3);
    push('a',stack,chk);
    push('b',stack,chk);
    stackPop(stack,chk);
    pop(stack,chk);
    stackPop(stack,chk);
    pop(stack,chk);
    destroyStack(stack);
    stackPop(stack,chk);

    return 0;
}
char* createStack(int n)
{
    char *stack;
    
    
    stack = malloc(sizeof(char) * n);
    printf("스택이 생성되었습니다\n");
    return stack;
}
void push(char target, char* stack, int* chk)
{
    //isfullstack
    stack[chk[0] + 1] = target;
    printf("%c이 삽입되었습니다\n",stack[chk[0] + 1]);
    chk[0] += 1;
}
void pop(char* stack, int* chk)
{
    //isemptystack
    char data;
    printf("%c이 제거되었습니다\n",stack[chk[0]]);
    data = stack[chk[0]];
    chk[0] -= 1;
}
void stackPop(char *stack, int* chk)
{
    //isemptystack
    char data;
    printf("%c입니다\n",stack[chk[0]]);
    
}
void destroyStack(char *stack)
{
    free(stack);
    printf("스택이 해제되었습니다\n");
}





