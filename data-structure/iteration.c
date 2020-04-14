#include <stdio.h>
int fib(int n);

int main(void)
{
    int result;
    result = fib(5);
    printf("결과: %d", result);
    return 0;
}
int fib(int n)
{
    int fir = 0;
    int sec = 1;
    int result = 0;

    for(int i = 0;i <= n-2; i++)
    {
        result = sec + fir;
        fir = sec;
        sec = result;
    }
    return result;
}