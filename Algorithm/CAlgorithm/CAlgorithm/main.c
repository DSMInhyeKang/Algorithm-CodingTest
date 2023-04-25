//
//  main.c
//  Algorithm
//
//  Created by 강인혜 on 2022/06/30.
//

#include <stdio.h>

int cnt;

int fact(int n)
{
    if(n < 1)
        return 1;
    
    return n*fact(n-1);
}


int main()
{
    int N, k;
    int height[10];
    int i;
    scanf("%d %d", &N, &k);
    
    for (i = 0; i < N; i++) {
        scanf("%d", &height[i]);
    }
    
    for (i = 0; i < N; i++) {
        if(height[i] <= k)
            cnt++;
    }
    
    printf("%d", fact(cnt));
    
    return 0;
}
