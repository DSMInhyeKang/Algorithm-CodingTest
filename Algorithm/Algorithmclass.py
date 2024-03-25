# 정다면체
N, M = map(int, input().split())

# dict = {n: 0 for n in range(1, N+M+1)}

# for i in range(1, N+1):
#     for j in range(1, M+1):
#         dict[i+j] += 1

# result = [ k for k, v in dict.items() if v == max(dict.values())]

# print(*result)

for i in range(M-N+1):
      print(N+i+1, end=" ")



# 소수(에라토스테네스 체)
N = int(input())
primes = [False] * 2 + [True] * (N - 1)

for i in range(2, int(N ** 0.5) + 1):
    if primes[i] == True:
        for j in range(i * i, N + 1, i):
            primes[j] = False

print(len([n for n in range(N+1) if primes[n]]))



# 요세푸스 문제 (알실)
import sys

N, K = map(int, sys.stdin.readline().split())
queue = [i for i in range(1, N+1)]
order = []
idx = 0

while queue:
    idx = (idx + K - 1) % len(queue)
    order.append(queue.pop(idx))

print('<' + ','.join(map(str, order)) + '>')




# 가장 큰 수
n, m = map(int, input().split())
n = list(map(int, str(n)))

stack = []

for i in n:
    while m > 0 and stack and i > stack[-1]:
        stack.pop()
        m -= 1

    stack.append(i)
    
while m > 0:
    stack.pop()
    m -= 1

print(''.join(map(str, stack)))



# 공주 구하기
N, K = map(int, input().split())

princes = [ n for n in range(1, N+1) ]
cnt = 0

while len(princes) > 1:
    cnt += 1
    if cnt == K:
        del princes[0]
        cnt = 0
    else:
        princes.append(princes[0])
        del princes[0]

print(princes)



# 괄호 (알실)
for i in range(int(input())):
    stack = []
    ans = "YES"

    for c in input():
        if c == '(':
            stack.append(c)
        else:
            if stack:
                stack.pop()
            else:
                ans = "NO"

    if stack: 
        ans = "NO"
    
    print(ans)



# 부분집합 구하기(DFS)
N = int(input())
visited= [False for _ in range(N+1)]

def dfs(n):
    if n == N+1:
        for i in range(1, N+1):
            if visited[i] == True:
                print(i, end=" ")
        print()
        return   
    else:
        visited[n] = True
        dfs(n+1)
        visited[n] = False
        dfs(n+1)

dfs(1)



# 합이 같은 부분집합(DFS : 아마존 인터뷰)
import sys

N = int(input())
num = list(map(int, input().split()))
totalSum = sum(num)

def dfs(v, sum):
    if sum > totalSum//2:  # 문제에서 요구하는 조건과 부합하지 않기 때문에 탐색할 필요 없음
        return
    
    if v == N:
        if sum == totalSum//2:
            print("YES")
            sys.exit(0)
    else:
        dfs(v+1, sum+num[v])
        dfs(v+1, sum)

dfs(0, 0)
print("NO")



# 바둑이 승차(DFS)
C, N = map(int, input().split())
weight = [int(input()) for _ in range(N)]
max = 0

def dfs(v, sum):
    global max

    if sum > C:
        return

    if v == N:
        if sum <= C and sum > max:
            max = sum
            return
    else:
        dfs(v+1, sum+weight[v])
        dfs(v+1, sum)

dfs(0, 0)
print(max)



#  중복순열 구하기
N, M = map(int, input().split())
cnt = 0
h = [0] * M

def dfs(v):
    global cnt
    
    if v == M:
        for i in range(M):
            print(h[i], end=" ")
        print()
        cnt += 1
    else:
        for j in range(1, N+1):
            h[v] = j
            dfs(v+1)

dfs(0)
print(cnt)



# 카드(알실)
