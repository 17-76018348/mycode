#include <stdio.h>
int fib(int n);

int main(void)
{
    int result;
    result = fib(5);
    printf("결과는: %d", result);
    return 0;
}

int fib(int n)
{
    if(n == 0) return 0;
    else if(n == 1) return 1;
    return (fib(n-1) + fib(n-2));
}

