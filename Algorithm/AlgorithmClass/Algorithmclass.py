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

def dfs(v, sum, tsum):  # sum = 사용할 지 안 할 지 결정해서 더한 값, tsum = 무작정 더한 값
    global max

    if sum + (sum(weight) - tsum) < max:
        return

    if sum > C: # 바운더리 넘어간 순간 종료
        return

    if v == N: # 제일 아래 레벨에 도달했을 때
        if max < sum:
            max = sum
    else:
        dfs(v+1, sum+weight[v], tsum+weight[v])
        dfs(v+1, sum, tsum+weight[v])

dfs(0, 0, 0)
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
        return
    else:
        for j in range(1, N+1):
            h[v] = j
            dfs(v+1)

dfs(0)
print(cnt)



# 카드(알실)



# 동전교환
N = int(input())
coins = list(map(int, input().split()))
coins.sort(reverse=True)
M = int(input())
res = 214700000

def dfs(l, sum):
    global res
    if l < res:
        return

    if sum > M:
        return
    
    if sum == M:
        if l < res:
            res = l
    else:
        for i in range(N):
            dfs(l+1, sum+coins[i])

dfs(0, 0)
print(res)



# 순열 구하기
N, M = map(int, input().split())
result = [0] * N
visited= [False for _ in range(N+1)]
cnt = 0

def dfs(v):
    global cnt

    if v == M:
        for i in range(M):
            print(result[i], end=' ')
        print()
        cnt += 1
        return
    else:
        for i in range(1, N+1):
            if not visited[i]:
                visited[i] = True
                result[v] = i
                dfs(v+1)
                visited[i] = False

dfs(0)
print(cnt)





# 최대 점수 구하기(DFS)
N, M = map(int, input().split())
scores = []
times = []
max = 0

for _ in range(N):
    s, t = map(int, input().split())
    scores.append(s)
    times.append(t)

def dfs(l, scoreSum, timeSum):
    global max

    if timeSum > M:
        return

    if l == N:
        if max < scoreSum:
            max = scoreSum
    else:
        dfs(l+1, scoreSum+scores[l], timeSum+times[l])
        dfs(l+1, scoreSum, timeSum)


dfs(0, 0, 0)
print("\n" + str(max))



# 후위 표기식



# 백설 공주와 일곱 난쟁이



# 휴가(DFS)
N = int(input())
t = []
p = []
max = 0

for _ in range(N):
    a, b = map(int, input().split())
    t.append(a)
    p.append(b)

def dfs(l, sum):
    global max

    if l > N:
        return

    if l == N:
        if max < sum:
            max = sum
    else:
        dfs(l+t[l], sum+p[l])
        dfs(l+1, sum)

dfs(0, 0)
print(max)



# 양팔저울(DFS)
K = int(input())
weights = list(map(int, input().split()))
visited = [False] * (sum(weights) + 1)
cnt = 0

def dfs(l, s):
    if l > K or s > sum(weights):
        return

    if l == K:
        if s > 0:
            visited[s] = True
    else:
        dfs(l+1, s+weights[l])
        dfs(l+1, s-weights[l])
        dfs(l+1, s)

dfs(0, 1)

for i in visited[1:]:
    if i == False:
        cnt += 1

print(cnt)



# 송아지 찾기(BFS : 상태트리탐색)
from collections import deque

S, E = map(int, input().split())
visited = [False] * 10001
distance = [0] * 10001

visited[S] = True
deq = deque()
deq.append(S)

while deq:
    n = deq.popleft()
    
    if n >= 10001:
        break

    for next in (n+1, n-1, n+5):
        if 0 <= next < 10001:
            if not visited[next]:
                deq.append(next)
                visited[next] = True
                distance[next] = distance[n] + 1

print(distance[E])



# 사과나무(BFS)
from collections import deque

N = int(input())
farm = [list(map(int, input().split())) for _ in range(N)]
dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]
visited = [[False] * N for _ in range(N)]
sum = 0

visited[N//2][N//2] = True
sum += farm[N//2][N//2]
deq = deque()
deq.append((N//2, N//2))
level = 0

while True:
    if level == N//2:
        break

    for i in range(len(deq)):
        ctr = deq.popleft()

        for j in range(4):
            x = ctr[0] + dx[j]
            y = ctr[1] + dy[j]

            if not visited[x][y]:
                visited[x][y] = True
                sum += farm[x][y]
                deq.append((x, y))
        
    level += 1

print(sum)



# 미로의 최단 거리 통로(BFS 활용)
from collections import deque

maze = [list(map(int, input().split())) for _ in range(7)]
dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]
visited = [[0] * 7 for _ in range(7)]

deq = deque()
deq.append((0, 0))
maze[0][0] = 1

while deq:
    ctr = deq.popleft()

    for i in range(4):
        x = ctr[0] + dx[i]
        y = ctr[1] + dy[i]

        if 0 <= x <= 6 and 0 <= y <= 6 and maze[x][y] == 0:
            maze[x][y] = 1
            visited[x][y] = visited[ctr[0]][ctr[1]] + 1
            deq.append((x, y))

if visited[6][6] == 0:
    print(-1)
else:
    print(visited[6][6])



# 단지 번호 붙이기(BFS)
from collections import deque

N = int(input())
map = [list(map(int, input())) for _ in range(N)]
dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]
result, deq = [], deque()
cnt = 0

for i in range(N):
    for j in range(N):
        if map[i][j] == 1:
            map[i][j] = 0
            deq.append((i, j))
            cnt = 1

            while deq:
                ctr = deq.popleft()

                for k in range(4):
                    x = ctr[0] + dx[k]
                    y = ctr[1] + dy[k]

                    if 0 <= x < N and 0 <= y < N and map[x][y] == 1:
                        map[x][y] = 0
                        deq.append((x, y))
                        cnt += 1
            
            result.append(cnt)

print(len(result))
print(*sorted(result), sep='\n')



# 섬나라 아일랜드(BFS 활용)
from collections import deque

N = int(input())
map = [list(map(int, input().split())) for _ in range(N)]
dx, dy = [-1, -1, 0, 1, 1, 1, 0, -1], [0, 1, 1, 1, 0, -1, -1, -1]
deq = deque()
cnt = 0

for i in range(N):
    for j in range(N):
        if map[i][j] == 1:
            map[i][j] = 0
            deq.append((i, j))

            while deq:
                ctr = deq.popleft()

                for k in range(8):
                    x = ctr[0] + dx[k]
                    y = ctr[1] + dy[k]

                    if 0 <= x < N and 0 <= y < N and map[x][y] == 1:
                        map[x][y] = 0
                        deq.append((x, y))

            cnt += 1

print(cnt)