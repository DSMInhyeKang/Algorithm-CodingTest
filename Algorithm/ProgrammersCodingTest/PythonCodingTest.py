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



# 3 x n 타일링(12902) - Lv.2
def solution(n):
    answer = 0
    dp = [3, 11]
    
    if n <= 4:  
        return dp[n//2 - 1]
    else:
        for i in range(2, n//2):
            dp.append((dp[i-1] * 4 - dp[i-2]) % 1000000007)
            
    return dp[-1]



# [PCCP 기출문제] 2번 / 석유 시추(250136) - Lv.2
from collections import deque

def solution(land):
    reclamed = [0] * len(land[0])
    dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]

    for i in range(len(land)):
        for j in range(len(land[0])):
            if land[i][j] == 1:
                ground = set([j])
                oil = 1
                land[i][j] = 0
                deq = deque([[i, j]])

                while deq:
                    a, b = deq.popleft()

                    for k in range(4):
                        x = a + dx[k]
                        y = b + dy[k]

                        if 0 <= x < len(land) and 0 <= y < len(land[0]) and land[x][y] == 1:
                            ground.add(y)
                            oil += 1
                            land[x][y] = 0
                            deq.append([x, y])

                for l in ground:
                    reclamed[l] += oil

    return max(reclamed)



# 멀리 뛰기(12914) - Lv.2
def solution(n):
    dp = [0] * (n+1)
    
    dp[0] = 1
    dp[1] = 2
    
    for i in range(2, n):
        dp[i] = dp[i-2] + dp[i-1]
	
    return dp[n-1] % 1234567



# [PCCP 기출문제] 3번 / 아날로그 시계(250135) - Lv.2
def solution(h1, m1, s1, h2, m2, s2):
    answer = 0
    start = h1 * 3600 + m1 * 60 + s1
    end = h2 * 3600 + m2 * 60 + s2  

    if start == 0 * 3600 or start == 12 * 3600:
        answer += 1

    while start < end:
        currentH = start / 120 % 360
        currentM = start / 10 % 360
        currentS = start * 6 % 360

        nextH = 360 if (start + 1) / 120 % 360 == 0 else (start + 1) / 120 % 360
        nextM = 360 if (start + 1) / 10 % 360 == 0 else (start + 1) / 10 % 360
        nextS = 360 if (start + 1) * 6 % 360 == 0 else (start + 1) * 6 % 360

        if currentS < currentH and nextS >= nextH:
            answer += 1
        if currentS < currentM and nextS >= nextM:
            answer += 1
        if nextS == nextH and nextH == nextM:
            answer -= 1

        start += 1
    
    return answer



# 귤 고르기(138476) - Lv.2
from collections import Counter

def solution(k, tangerine):
    counter = Counter(tangerine)
    tangerine.sort(key = lambda t: (-counter[t], t))
    
    return len(set(tangerine[:k]))



# n^2 배열 자르기(87390) - Lv.2
def solution(n, left, right):
    answer = []
    
    for i in range(left, right + 1):
        answer.append(max(i//n, i%n) + 1)
        
    return answer



# 2개 이하로 다른 비트(77885) - Lv.2
def solution(numbers):
    answer = []

    for number in numbers:
        bin_number = list('0' + bin(number)[2:])
        idx = ''.join(bin_number).rfind('0')
        bin_number[idx] = '1'
        
        if number % 2 == 1:
            bin_number[idx+1] = '0'
        
        answer.append(int(''.join(bin_number), 2))

    return answer

    # 생각지도 못한 풀이
    def solution(numbers):
        answer = []
        
        for idx, val in enumerate(numbers):
          answer.append(((val ^ (val+1)) >> 2) +val +1)

        return answer



# 괄호 변환(60058) - Lv.2
def solution(p):
    if not p :
        return p

    r, c = True, 0
    
    for i in range(len(p)):
        if p[i] == '(':
            c -= 1
        else:
            c += 1
            
        if c > 0: 
            r = False
        
        if c == 0:
            if r:
                return p[:i + 1] + solution(p[i + 1:])
            else:
                return '(' + solution(p[i + 1:]) + ')' + ''.join(list(map(lambda x: '(' if x == ')' else ')', p[1:i])))
            


# 순위 검색(72412) - Lv.2
def solution(info, query):
    data = dict()
    
    for a in ['cpp', 'java', 'python', '-']:
        for b in ['backend', 'frontend', '-']:
            for c in ['junior', 'senior', '-']:
                for d in ['chicken', 'pizza', '-']:
                    data.setdefault((a, b, c, d), list())

    for i in info:
        i = i.split()
        
        for a in [i[0], '-']:
            for b in [i[1], '-']:
                for c in [i[2], '-']:
                    for d in [i[3], '-']:
                        data[(a, b, c, d)].append(int(i[4]))

    for k in data:
        data[k].sort()

    answer = list()

    for q in query:
        q = q.split()
        scores = data[(q[0], q[2], q[4], q[6])]
        wanted = int(q[7])
        l, r = 0, len(scores)
        
        while l < r:
            middle = (l + r) // 2

            if scores[middle] >= wanted:
                r = middle
            else:
                l = middle + 1
                
        answer.append(len(scores)-l)
        
    return answer



# [PCCP 기출문제] 1번 / 붕대 감기(242258) - Lv.1
def solution(bandage, health, attacks):
    t, x, y = bandage
    hp = health
    time = 0

    for sec, damage in attacks:
        gap = sec - time - 1
        hp = min(hp + y * (gap // t) + x * gap, health)
        hp -= damage

        if hp < 1:
            return -1
        
        time = sec

    return hp



# BOJ - N-Queen(9633)
n, ans = int(input()), 0
a, b, c = [False] * n, [False] * (2*n-1), [False] * (2*n-1)

def backtracking(i):
    global ans
    
    if i == n:
        ans += 1
        return
    
    for j in range(n):
        if not (a[j] or b[i+j] or c[i-j+n-1]):
            a[j] = b[i+j] = c[i-j+n-1] = True
            backtracking(i+1)
            a[j] = b[i+j] = c[i-j+n-1] = False


backtracking(0)
print(ans)



# 숫자 변환하기(154538) - Lv.2
from collections import deque

def solution(x, y, n):
    queue = deque()
    queue.append((x, 0))
    visited = set()
    
    while queue:
        i, j = queue.popleft()
        
        if i > y or i in visited:
            continue
            
        visited.add(i)
        
        if i == y: 
            return j
        
        for k in (i*3, i*2, i+n):
            if k <= y and k not in visited:
                queue.append((k, j+1))    
                
    return -1



# 침몰하는 타이타닉
from collections import deque

n, m = map(int, input().split())
people = list(map(int, input().split()))
people.sort()
people = deque(people)
cnt = 0

while p :
    if len(people) == 1:
        cnt += 1
        break

    if people[0] + people[-1] > m:
        people.pop()
        cnt += 1
    else :
        people.popleft()
        people.pop()
        cnt += 1

print(cnt)



# 혼자서 하는 틱택토(160585) - Lv.2
def solution(board):
    strboard = ''.join(board)
    valid = strboard.count('O') - strboard.count('X')
    colboard = list(zip(*board))
    oCnt, xCnt = 0, 0
    
    if valid not in [0, 1]:
        return 0
    
    for i in range(3):
        if colboard[i].count('O') == 3 or board[i].count('O') == 3:
            oCnt += 1
        if colboard[i].count('X') == 3 or board[i].count('X') == 3:
            xCnt += 1
    
    for i in range(0, 3, 2):
        if (board[0][i] == board[1][1] == board[2][2-i] == 'O'):
            oCnt += 1
        if (board[0][i] == board[1][1] == board[2][2-i] == 'X'):
            xCnt += 1
    
    if oCnt and xCnt:
        return 0
    if oCnt == 1 and valid == 0:
        return 0
    if xCnt == 1 and valid >= 1:
        return 0
    
    return 1