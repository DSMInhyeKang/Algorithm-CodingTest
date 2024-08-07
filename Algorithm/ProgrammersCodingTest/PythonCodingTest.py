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



# n + 1 카드게임(258707) - Lv.3
import heapq

def solution(coin, cards):
    assert isinstance(cards, list)
    n = len(cards)
    used, q = [], []
    step, idx = 0,  int(n / 3)

    for i in range(int(n / 3)):
        used.append(cards[i])

        if n + 1 - cards[i] in used and cards.index(n - cards[i] + 1) < int(n / 3):
            heapq.heappush(q, [0, cards[i], n - cards[i] + 1])

    while True:
        step += 1

        if idx >= n:
            break

        for _ in range(2):
            used.append(cards[idx])
            if n - cards[idx] + 1 in used:
                if cards.index(n - cards[idx] + 1) < int(n / 3):
                    heapq.heappush(q, [1, cards[idx], n - cards[idx] + 1])
                else:
                    heapq.heappush(q, [2, cards[idx], n - cards[idx] + 1])
            idx += 1

        if not q:
            break

        c, card1, card2 = heapq.heappop(q)

        if coin >= c:
            coin -= c
        else:
            break

    return step



# 주사위 고르기(258709) - Lv.3
from itertools import combinations, product
from bisect import bisect_left

def solution(dices):
    dic = {}
    L = len(dices)
    
    for a in combinations(range(L), L//2):
        b = [i for i in range(L) if i not in a]
        A, B = [], []
        
        for order in product(range(6), repeat=L//2):
            A.append(sum(dices[i][j] for i, j in zip(a, order)))
            B.append(sum(dices[i][j] for i, j in zip(b, order)))
        B.sort()

        wins = sum(bisect_left(B, num) for num in A)
        dic[wins] = list(a)

    maxKey = max(dic.keys())

    return [x+1 for x in dic[maxKey]]



# 아방가르드 타일링(181186) - Lv.3
def solution(n):
    dp = [1, 1, 3, 10] + [0] * (n-3)
    sp_c2, sp_c3 = 2, 5
    sp_cases = [12, 2, 4]  # 각각 6, 4, 5 일 때 초기에 더해지는 값들
    
    if n <= 3: return dp[n]
    
    for idx in range(4, n+1):
        sp_c = idx % 3
        total = sp_cases[sp_c]
        
        total += dp[idx-1] + sp_c2 * dp[idx-2] + sp_c3 * dp[idx-3]
        dp[idx] = total
        
        sp_cases[sp_c] += dp[idx-1] * 2 + dp[idx-2] * 2 + dp[idx-3] * 4
        
    answer = dp[n] % 1000000007
    
    return answer



# 롤케이크 자르기(132265) - Lv.2
from collections import Counter

def solution(topping):
    dic = Counter(topping)
    set_dic = set()
    res = 0
    
    for i in topping:
        dic[i] -= 1
        set_dic.add(i)
        
        if dic[i] == 0: dic.pop(i)
            
        if len(dic) == len(set_dic): res += 1
            
    return res



# 에어컨(214289) - Lv.3
def solution(temperature, t1, t2, a, b, onboard):
    k = 1000 * 100
    t1 += 10
    t2 += 10
    temperature += 10
    
    dp = [[k] * 51 for _ in range(len(onboard))]
    dp[0][temperature] = 0
    
    flag = 1
    if temperature > t2:
        flag = -1
        
    for i in range(1, len(onboard)):
        for j in range(51):
            arr = [k]
            
            if (onboard[i] == 1 and t1 <= j <= t2) or onboard[i] == 0:
                if 0 <= j+flag <= 50:
                    arr.append(dp[i-1][j+flag])
                if j == temperature:
                    arr.append(dp[i-1][j])
                if 0 <= j-flag <= 50:
                    arr.append(dp[i-1][j-flag] + a)
                if t1 <= j <= t2:
                    arr.append(dp[i-1][j] + b)

                dp[i][j] = min(arr)
            
    answer = min(dp[len(onboard)-1])
    
    return answer



# [PCCP 기출문제] 4번 / 수레 움직이기(250134) - Lv.3
import heapq

def solution(maze):
    answer = 0

    redX, redY = 0, 0
    blueX, blueY = 0, 0
    redEndX, redEndY = 0, 0
    blueEndX, blueEndY = 0, 0
    redVisited , blueVisited = [], []

    for row in range(len(maze)):
        for col in range(len(maze[0])):
            if maze[row][col] == 1:
                redX, redY = row, col
                redVisited.append((row, col))
            elif maze[row][col] == 2:
                blueX, blueY = row, col
                blueVisited.append((row, col))
            elif maze[row][col] == 3:
                redEndX, redEndY = row, col
            elif maze[row][col] == 4:
                blueEndX, blueEndY = row, col
            elif maze[row][col] == 5:
                redVisited.append((row, col))
                blueVisited.append((row, col))
            else:
                continue
    
    answer = bfs(redX, redY, blueX, blueY, redEndX, redEndY, blueEndX, blueEndY, maze, redVisited, blueVisited)
    return answer

def bfs(rx, ry, bx, by, rex, rey, bex, bey, maze, redVisited, blueVisited):
    q = []
    direction = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    heapq.heappush(q, (0, rx, ry, bx, by, redVisited, blueVisited))

    while q:
        cnt, crx, cry, cbx, cby, _redVisited, _blueVisited = heapq.heappop(q)
        redArrive = False
        blueArrive = False

        if (crx, cry) == (cbx, cby):
            continue
        if not redArrive and (crx, cry) == (rex, rey):
            redArrive = True
        if not blueArrive and (cbx, cby) == (bex, bey):
            blueArrive = True
            
        if redArrive and blueArrive:
            return cnt
        elif redArrive:
            for dx, dy in direction:
                nbx, nby = cbx + dx, cby + dy

                if not (0 <= nbx < len(maze) and 0 <= nby < len(maze[0])):
                    continue
                if not (nbx, nby) in _blueVisited:
                    _blueVisited.append((nbx, nby))
                    heapq.heappush(q, (cnt + 1, crx, cry, nbx, nby, _redVisited[:], _blueVisited[:]))
                    _blueVisited.pop()
        elif blueArrive:
            for dx, dy in direction:
                nrx, nry = crx + dx, cry + dy

                if not (0 <= nrx < len(maze) and 0 <= nry < len(maze[0])):
                    continue
                if not (nrx, nry) in _redVisited:
                    _redVisited.append((nrx, nry))
                    heapq.heappush(q, (cnt + 1, nrx, nry, cbx, cby, _redVisited[:], _blueVisited[:]))
                    _redVisited.pop()
        else:
            for dbx, dby in direction:
                nbx, nby = cbx + dbx, cby + dby

                if not (0 <= nbx < len(maze) and 0 <= nby < len(maze[0])):
                    continue
                if (nbx, nby) in _blueVisited:
                    continue
                _blueVisited.append((nbx, nby))

                for drx, dry in direction:
                    nrx, nry = crx + drx, cry + dry
                    
                    if (nrx, nry) == (cbx, cby) and (nbx, nby) == (crx, cry):
                        continue
                    if not (0 <= nrx < len(maze) and 0 <= nry < len(maze[0])):
                        continue
                    if not (nrx, nry) in _redVisited:
                        _redVisited.append((nrx, nry))
                        heapq.heappush(q, (cnt + 1, nrx, nry, nbx, nby, _redVisited[:], _blueVisited[:]))
                        _redVisited.pop()

                _blueVisited.pop()

    return 0



# 주차 요금 계산(92341) - Lv.2
from datetime import datetime
import math

def solution(fees, records):
    answer = { record.split()[1]:0 for record in records }
    car = {}
    
    for record in records:
        time, car_num, inout = record.split()

        if inout == 'IN':
            car[car_num] = datetime.strptime(time,"%H:%M")
        else:
            pay = str(datetime.strptime(time,"%H:%M") - car[car_num])
            pay = int(pay.split(':')[0]) * 60 + int(pay.split(':')[1])
            answer[car_num] += pay
            del car[car_num]

    if len(car) > 0:
        for i in car:
            pay = str(datetime.strptime('23:59',"%H:%M") - car[i])
            pay = int(pay.split(':')[0]) * 60 + int(pay.split(':')[1])
            answer[i] += pay

    answer = sorted(answer.items())
    
    for i in range(len(answer)):
        if answer[i][1] <= fees[0]:
            answer[i] = fees[1]
        else:
            answer[i] = fees[1] + math.ceil((answer[i][1] - fees[0])/fees[2]) * fees[3]

    return answer

# Class + Lambda 활용 풀이
from collections import defaultdict
from math import ceil

class Parking:
    def __init__(self, fees):
        self.fees = fees
        self.in_flag = False
        self.in_time = 0
        self.total = 0

    def update(self, t, inout):
        self.in_flag = True if inout=='IN' else False
        if self.in_flag:
            self.in_time = str2int(t)
        else:
            self.total  += (str2int(t)-self.in_time)

    def calc_fee(self):
        if self.in_flag: self.update('23:59', 'out')
        add_t = self.total - self.fees[0]
        return self.fees[1] + ceil(add_t/self.fees[2]) * self.fees[3] if add_t >= 0 else self.fees[1]

def str2int(string):
    return int(string[:2])*60 + int(string[3:])

def solution(fees, records):
    recordsDict = defaultdict(lambda:Parking(fees))
    for rcd in records:
        t, car, inout = rcd.split()
        recordsDict[car].update(t, inout)
    return [v.calc_fee() for k, v in sorted(recordsDict.items())]



# 디펜스 게임(142085) - Lv.2
import heapq as hq

def solution(n, k, enemy):
    q = enemy[:k]
    hq.heapify(q)

    for idx in range(k,len(enemy)):
        n -= hq.heappushpop(q, enemy[idx])

        if n < 0:
            return idx
        
    return len(enemy)



# 의상(42578) - Lv.2
def solution(clothes):
    closet = {} 
    
    for name, kind in clothes:
        if kind in closet.keys():
            closet[kind] += [name]
        else:
            closet[kind] = [name]
    
    answer = 1
    
    for _, value in closet.items():
        answer *= (len(value) + 1)
        
    return answer -1


# Counter 활용 풀이
from collections import Counter
from functools import reduce

def solution(clothes):
    cnt = Counter([kind for name, kind in clothes])
    answer = reduce(lambda x, y: x*(y+1), cnt.values(), 1) - 1
    return answer



# 정수 삼각형(43105) - Lv.2
def solution(triangle):
    dp = triangle
    
    for i in range(1, len(triangle)):
        for j in range(i+1):
            if j == 0:
                dp[i][j] += dp[i-1][j]
            elif i == j:
                dp[i][j] += dp[i-1][j-1]
            else:
                dp[i][j] += max(dp[i-1][j], dp[i-1][j-1])
                
    return max(dp[len(triangle)-1])



# BOJ - 치킨 배달(15686): G5
from itertools import combinations

N, M = map(int, input().split())
city = [list(map(int, input().split())) for _ in range(N)]
house, chicken = [], []
total = 250000

for i in range(N):
    for j in range(N):
        if city[i][j] == 1:
            house.append((i, j))
        elif city[i][j] == 2:
            chicken.append((i, j))

for c in combinations(chicken, M):
    cityDis = 0

    for h in house:
        d = 2500

        for r in range(M):
            d = min(d, abs(h[0]-c[r][0]) + abs(h[1]-c[r][1]))
        
        cityDis += d

    total = min(total, cityDis)

print(total)



# BOJ - 연산자 끼워넣기(14888): S1
N = int(input())
A = list(map(int, input().split()))
op = list(map(int, input().split()))  # +, -, *, //
maximum, minimum = -1e9, 1e9

def dfs(l, t, a, s, m, d):
    global maximum, minimum

    if l == N:
        maximum = max(t, maximum)
        minimum = min(t, minimum)
        return

    if a:
        dfs(l+1, t+A[l], a-1, s, m, d)
    if s:
        dfs(l+1, t-A[l], a, s-1, m, d)
    if m:
        dfs(l+1, t*A[l], a, s, m-1, d)
    if d:
        dfs(l+1, int(t/A[l]), a, s, m, d-1)

dfs(1, A[0], op[0], op[1], op[2], op[3])
print(maximum)
print(minimum)



### [Toyota Programming Contest 2024#8（AtCoder Beginner Contest 365）]

# A - Leap Year :: AC(100)
Y = int(input())

if (Y % 4 == 0 and Y % 100 != 0) or (Y % 400 == 0):
    print(366)
else:
    print(365)



# B - Second Best :: AC(200)
N = int(input())
seq = list(map(int, input().split()))
second = sorted(seq, reverse=True)[1]
print(seq.index(second)+1)



# C - Transportation Expenses
## 앞으로 CPython과 PyPy 중에 무조건 PyPy로 돌려야겠다..^^ 
## CPython에서는 TLE(시간 초과)  PyPy에서는 AC(정답) ㅠㅠ
import sys

N, M = map(int, sys.stdin.readline().split())
people = list(map(int, sys.stdin.readline().split()))
s, e = 1, M
x = 0

if sum(people) <= M:
    print('infinite')
    sys.exit()

while s <= e:
    mid = (s + e) // 2
    total = 0
    if mid < x: break

    for p in people:
        total += min(mid, p)

        if total > M: break
    
    if total <= M:
        x = mid
        s = mid + 1
    else:
        e = mid - 1

print(x)


## Accepted Correctly
import sys

N, M = map(int, sys.stdin.readline().split())
people = list(map(int, sys.stdin.readline().split()))
s, res, cum = sum(people), 0, 0
x = 0

if sum(people) <= M:
    print('infinite')
    sys.exit()

for i, a in enumerate(sorted(people)):
    if cum + a * (N - i) > M:
        x = (M - cum) // (N - i)
        break

    cum += a

print(x)



# E - Xor Sigma Problem :: TLE
def combine_arrays(A):
    N = len(A)
    result = 0

    cumulative_xor = [0] * (N + 1)

    for i in range(N):
        cumulative_xor[i + 1] = cumulative_xor[i] ^ A[i]

    for i in range(N):
        for j in range(i + 1, N):
            combined = cumulative_xor[j + 1] ^ cumulative_xor[i]  # A[i]부터 A[j]까지의 XOR
            result += combined

    return result

N = int(input())
A = list(map(int, input().split()))
print(combine_arrays(A))


## Accepted Correctly
N = int(input())
A = list(map(int, input().split()))
l = 30  # the length of bit
ans = -sum(A)

for p in range(l):  # current bit position
    x = 0  # xor result
    c = [1, 0]  # cnt. c[0]: p == 0, c[1]: p == 1

    for a in A:
        x ^= (a >> p) & 1
        c[x] += 1

    ans += c[0] * c[1] << p

print(ans)



# D - AtCoder Janken 3
N = int(input())
S = input()

dp = [0] * 3
s = "RSP"

for i in range(N):
    u = s.index(S[i])
    v = (u - 1) % 3
    ndp = [0] * 3

    ndp[u] = max(dp[(u+1)%3], dp[(u+2)%3])
    ndp[v] = max(dp[(v+1)%3], dp[(v+2)%3]) + 1

    dp = list(ndp)

print(max(dp))



# 도둑질(42897) - Lv.4
def solution(money):
    dp1 = [0] * len(money)
    dp2 = [0] * len(money)
    
    dp1[0] = money[0]
    
    for i in range(1, len(money) - 1):
        dp1[i] = max(dp1[i - 1], dp1[i - 2] + money[i])

    dp1[0] = 0
    
    for i in range(1, len(money)):
        dp2[i] = max(dp2[i - 1], dp2[i - 2] + money[i])

    return max(dp1[-2], dp2[-1])