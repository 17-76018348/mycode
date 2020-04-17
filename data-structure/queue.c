#include <stdio.h>
#include <stdlib.h>
typedef int Element;
typedef struct tQueueNode
{
    Element data;
    struct queueNode *next;
} QueueNode;
typedef struct{
    int count;
    QueueNode *front, *rear;
} Queue;
int main(void)
{
    return 0;    
}
Queue* CreateQueue(int size)
{
    Queue *pNewQueue = (Queue*)malloc(sizeof(Queue));
    if(pNewQueue == NULL)
    {
        return NULL;
    }
    pNewQueue->count = 0;
    pNewQueue->front = pNewQueue->rear = NULL;

    return pNewQueue;
}
void Enqueue(Queue *pQueue, Element item)
{
    QueueNode *pNewNode = (QueueNode*)malloc(sizeof(QueueNode));
    if(pNewNode == NULL)
        return;
    pNewNode->data = item;
    pNewNode->next = NULL;

    if(pQueue->count <= 0)
    {
        pQueue->front = pQueue->rear = pNewNode;
    }
    else
    {
        pQueue->rear->next = pNewNode;
        pQueue->rear = pNewNode;
    }
    pQueue->count++;
    
}
Element Dequeue(Queue *pQueue)
{
    QueueNode *pFront = NULL;
    Element item = 0;

    if(pQueue->count <= 0)
        return 0;
    
    pFront = pQueue->front;
    item = pFront->data;
    if(pQueue->count == 1){
        pQueue->front = pQueue->rear = NULL;
    }
}