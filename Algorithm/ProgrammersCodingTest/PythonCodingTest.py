# [PCCP 기출문제] 1번 / 붕대 감기(250137) - Lv.1
def solution(bandage, health, attacks):
    hp = health 
    time = 0
    bonus = 0
    
    for i in range(len(attacks)):
        if hp <= 0:
            return -1
        
        while True:
            time += 1
            bonus += 1
            
            if attacks[i][0] == time:
                hp -= attacks[i][1]
                bonus = 0
                break
            else:
                if hp != health:
                    if bonus == bandage[0]:
                        hp += (bandage[1] + bandage[2])
                        bonus = 0
                    else:
                        hp += bandage[1]
                    
                    if hp > health:
                        hp = health    
            
    return hp if hp > 0 else -1



# [PCCE 기출문제] 10번 / 데이터 분석(250121) - Lv.1
def solution(data, ext, val_ext, sort_by):
    answer = []
    by = [ "code", "date", "maximum", "remain" ]

    for item in data:
        if item[by.index(ext)] < val_ext:
            answer.append(item)

    return sorted(answer, key=lambda x: x[by.index(sort_by)])



# BOJ - DFS와 BFS(1360)
N, M, V = map(int,input().split())

graph = [[False]*(N+1) for _ in range(N+1)]
for i in range (M):
    a,b = map(int,input().split())
    graph[a][b] = graph[b][a] = True

dfs_visited = [False]*(N+1)
bfs_visited = [False]*(N+1)

def dfs(V):
    dfs_visited[V] = True
    print(V, end=' ')
    for i in range(1, N+1):
        if graph[V][i] == True and dfs_visited[i] == False:
            dfs(i)

def bfs(V):
    queue = [V]
    bfs_visited[V] = True
    while queue:
        V = queue.pop(0)
        print(V, end = ' ')
        for i in range(1, N+1):
            if(bfs_visited[i] == False and graph[V][i] == True):
                queue.append(i)
                bfs_visited[i] = True

dfs(V)
print()
bfs(V)



# BOJ - 연결 요소의 개수(11724)
import sys
sys.setrecursionlimit(10**7)  # 백준에서 이거 안 돌리면 런타임 에러
input = sys.stdin.readline

N, M = map(int, input().split())
graph = [[] for _ in range(N+1)]

for _ in range(M):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

cnt = 0
visited = [False] * (N+1)

def dfs(graph, v, visited):
    visited[v] = True

    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)

for i in range(1, N+1):
    if not visited[i]:
        dfs(graph, i, visited)
        cnt += 1

print(cnt)



# BOJ - 결혼식(5567)
import sys 
from collections import deque

n=int(input())
m=int(input())
relation=[[] for _ in range(n+1)]

for _ in range(m):
    a,b=map(int, input().split())
    relation[a].append(b)
    relation[b].append(a)

res=[]

for friend in relation[1]:
    res.append(friend)
    res += relation[friend]

ans = list(set(res))
    
if len(ans)==0:
    print(len(ans))
else:
    print(len(ans)-1)



# BOJ - 퇴사(14501)
n = int(input())
t = []
p = []
max = 0

for _ in range(n):
    a, b = map(int, input().split())
    t.append(a)
    p.append(b)

def dfs(l, sum):
    global max

    if l > n:
        return
    
    if l == n:
        if sum > max:
            max = sum
    else:
        dfs(l+t[l], sum+p[l])
        dfs(l+1, sum)

dfs(0, 0)
print(max)



# BOJ - 양팔저울(17610)
n = int(input())
weights = list(map(int, input().split()))
weightSum = sum(weights)
res = set()

def dfs(l, sum):
    global res

    if l == n:
        if 0 < sum <= weightSum:  # 음수가 나와도 어차피 다른 가지에서 똑같이 양수 나옴
            res.add(sum)
    else:
        dfs(l+1, sum+weights[l])
        dfs(l+1, sum-weights[l])
        dfs(l+1, sum)

dfs(0,0)
print(weightSum - len(res))  # 전체 중에서 저울로 판별 가능한 수 뺌



# 동전 바꿔주기(DFS)
T = int(input())
k = int(input())
p = []
n = []
cnt = 0

for _ in range(k):
    pi, ni = map(int, input().split())
    p.append(pi)
    n.append(ni)

def dfs(l, sum):
    global cnt

    if sum > T or l > k:
        return
    
    if l == k:
        if sum == T:
            cnt += 1
    else:
        for i in range(n[l]+1):
            dfs(l+1, sum+(p[l]*i))

dfs(0, 0)
print(cnt)



# 동전 분배하기(DFS)
N = int(input())
coins = [int(input()) for _ in range(N)]
money = [0, 0, 0]
res = 2147000000

def dfs(l):
    global res

    if l == N:
        dif = (max(money) - min(money))

        if dif < res:
            temp = set()

            for i in money:
                temp.add(i)

            if len(temp) == 3:
                res = (max(money) - min(money))
    else:
        for i in range(3):
            money[i] += coins[l]
            dfs(l+1)
            money[i] -= coins[l]

dfs(0)
print(res)



# 알파코드(DFS)
code = list(map(int, input().split()))
n = len(code)
res = [0] * (n+3)
cnt = 0

def dfs(l, p):
    global cnt

    if l == n:
        cnt += 1

        for j in range(p):
            print(chr(res[j]+64), end='')
        print()
    else:
        for i in range(1, 27):
            if code[l] == i:
                res[p] = i
                dfs(l+1, p+1)
            elif i >= 10 and code[l] == (i//10) and code[l+1] == (i%10):
                res[p] = i
                dfs(l+2, p+1)


code.insert(n, -1)
dfs(0, 0)
print(cnt)



# 미로 탐색(DFS)



# 등산 경로(DFS)



# BOJ - 단지 번호 붙이기(2667)
N = int(input())
map = [list(map(int, input())) for _ in range(N)]
dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]
res = []

def dfs(x, y):
    global cnt
    cnt += 1
    map[x][y] = 0

    for i in range(4):
        xx = x + dx[i]
        yy = y + dy[i]

        if 0 <= xx < N and 0 <= yy < N and map[xx][yy] == 1:
            dfs(xx, yy)

for i in range(N):
    for j in range(N):
        if map[i][j] == 1:
            cnt = 0
            dfs(i, j)
            res.append(cnt)

print(len(res))
res.sort()
for x in res:
    print(x)


# 안전영역(DFS)



# 사다리 타기(DFS)



# 피자배달거리(DFS)



# 조합 구하기(DFS)
N, M = map(int, input().split())
res = [0] * M
cnt = 0

def dfs(l, start):
    global cnt 
    if l == M:
        for i in res:
            print(i, end=' ')
        print()
        cnt += 1
        return
    else:
        for i in range(start, N+1):
            res[l] = i
            dfs(l+1, i+1)

dfs(0, 1) # 1부터 N까지 중에 M개
print(cnt)



# 타겟 넘버(43165) - Lv.2
def solution(numbers, target):
    leaves = [0]    
    count = 0 

    for n in numbers : 
        temp = []
	
        for l in leaves : 
            temp.append(l + n)
            temp.append(l - n)
        
        leaves = temp 

    for l in leaves : 
        if l == target :
            count += 1
    
    return count



# 도넛과 막대 그래프(258711) - Lv.2
from collections import defaultdict

def solution(edges):
    answer = [0, 0, 0, 0]
    dic = defaultdict(lambda: [0, 0])

    for o, i in edges:
        dic[o][0] += 1  # 나간 간선
        dic[i][1] += 1  # 들어온 간선
        
    for v, item in dic.items():
        outEdge = item[0]
        inEdge = item[1]
        
        if outEdge >= 2 and inEdge == 0:  # 생성한 정점
            answer[0] = v
        elif outEdge == 0 and inEdge >= 1:  # 막대 모양
            answer[2] += 1
        elif outEdge >= 2 and inEdge >= 2:  # 8자 모양
            answer[3] += 1

    answer[1] = dic[answer[0]][0] - answer[2] - answer[3]
    
    return answer



# 과제 진행하기(176962) - Lv.2
def solution(plans):
    plans = sorted(map(lambda x: [x[0], int(x[1][:2]) * 60 + int(x[1][3:]), int(x[2])], plans), key=lambda x: -x[1])

    lst = []
    while plans:
        x = plans.pop()
        for i, v in enumerate(lst):
            if v[0] > x[1]:
                lst[i][0] += x[2]
        lst.append([x[1] + x[2], x[0]])
    lst.sort()

    return list(map(lambda x: x[1], lst))



# 2 x n 타일링(12900) - Lv.2
def solution(n):
    dp = [0 for i in range(n)]
    
    dp[0], dp[1] = 1, 2
    
    for i in range(2, n):
        dp[i] = (dp[i-1] + dp[i-2]) % 1000000007

    return dp[n-1]



# 124 나라의 숫자(12899) - Lv.2
def solution(n):
    arr = ['1','2','4']
    ans = ""

    while n > 0 :
        n = n-1
        ans = arr[n%3] + ans
        n //= 3

    return ans