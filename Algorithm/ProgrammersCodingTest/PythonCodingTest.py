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



# BOJ - 구간 합 구하기 4(11659): S3
import sys
input = sys.stdin.readline

N, M = map(int, input().spliat())
numbers = list(map(int, input().split()))
prefixSum = [0]
tmp = 0

for i in numbers:
    tmp += i
    prefixSum.append(tmp)

for _ in range(M):
    i, j = map(int, input().split())
    print(prefixSum[j] - prefixSum[i-1])



### AtCoder Beginner Contest 366

# A - Election 2 :: AC(100)
N, T, A = map(int, input().split())
Takahashi, Aoki = N-A,  N-T
print('Yes') if Aoki < T or Takahashi < A else print('No')

# 이렇게도 풀 수 있음
print('Yes') if T > N / 2 or A > N / 2 else print('No')



# B - Vertical Writing :: X
# 어떻게 풀 지 생각하다가 포기하고 다른 문제로 넘어감
def vertical_text_conversion(strings):
    max_length = max(len(s) for s in strings)
    N = len(strings)

    result = [''] * max_length

    for j in range(max_length):
        for i in range(N):
            if j < len(strings[i]):
                result[j] += strings[i][j]
            else:
                result[j] += '*'

    return result

N = int(input())
S = [list(input()) for _ in range(N)][::-1]

converted_strings = vertical_text_conversion(S)
for line in converted_strings:
    print(line)


## Accepted Correctly
n = int(input())
sss = [input() + '*' * 100 for _ in range(n)]
sss = list(''.join(reversed(s)) for s in zip(*sss))
ng = '*' * n
for row in sss:
    if row == ng:
        break
    print(row.rstrip('*'))



# C - Balls and Bag Query :: AC(300)
from collections import defaultdict

Q = int(input())
queries = [list(map(int, input().split())) for _ in range(Q)]
bag = defaultdict(int)

for q in queries:
    if q[0] == 1:
        bag[q[1]] += 1
    elif q[0] == 2:
        bag[q[1]] -= 1
        if bag[q[1]] == 0:
            del bag[q[1]]
    else:
        print(len(bag))



# E - Manhattan Multifocal Ellipse :: X
N, D = map(int, input().split())
points = [list(map(int, input().split())) for _ in range(N)]
points.sort(key=lambda x: x[0])

s, e = sorted(points)[0] - D, sorted(points)[-1]

print(s, e)

def manhattan_distance(A, B):
  distance = 0
  for i in range(len(A)):
    distance += abs(A[i] - B[i])
  return distance

print(manhattan_distance(A=[1, 5, 7, 9], B=[2, 3, 6, 15]))


## Accepted Correctly
def solve(lst):
    res = [0] * (D+1)
    cnt, idx = 0, 0
    lst.sort()
    x, d = -M, sum(abs(x-i) for i in lst)

    while x <= M:
        if d <= D: res[d] += 1

        x += 1
        d -= (n-2*cnt)

        while idx < n and lst[idx] == x:
            cnt += 1
            idx += 1

    return res

M = 2*10**6
n,D = map(int, input().split())
x = [0]*n
y = [0]*n
for i in range(n):
    x[i],y[i] = map(int, input().split())

rx = solve(x)
ry = solve(y)
for i in range(1,len(rx)):
    rx[i] += rx[i-1]

ans = 0
for i in range(D+1):
    ans += ry[i]*rx[D-i]
print(ans)



# BOJ - 스타트와 링크(14889): S1
N = int(input())
S = [list(map(int, input().split())) for _ in range(N)]
visited = [False for _ in range(N)]
res = float('inf')

def dfs(d, l):
    global res
    
    if d == N // 2:
        start, link = 0, 0
        
        for i in range(N):
            for j in range(N):
                if visited[i] and visited[j]:
                    start += S[i][j]
                elif not visited[i] and not visited[j]:
                    link += S[i][j]
                    
        res = min(res, abs(start-link))
        
        return
    else:
        for k in range(l, N):
            if not visited[k]:
                visited[k] = True
                dfs(d+1, k+1)
                visited[k] = False
                
dfs(0, 0)
print(res)



# BOJ - 마법사 상어와 비바라기(21610): G5
import sys

N, M = map(int, sys.stdin.readline().split())
graph = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
directions = [list(map(int, sys.stdin.readline().split())) for _ in range(M)]

rx = [0, -1, -1, -1, 0, 1, 1, 1]
ry = [-1, -1, 0, 1, 1, 1, 0, -1]
clouds = [(N-1, 0), (N-1, 1), (N-2, 0), (N-2, 1)]

for direction in directions:
    d, m = direction[0] - 1, direction[1] % N
    not_cloud = set()
    
    while clouds:
        x, y = clouds.pop()
        nx, ny = (x + m*rx[d]) % N, (y + m*ry[d]) % N
        graph[nx][ny] += 1
        not_cloud.add((nx, ny))

    for nx, ny in not_cloud:
        count = 0
        
        for _ in range(1, 8, 2):
            nnx = nx + rx[_]
            nny = ny + ry[_]
            
            if 0 <= nnx < N and 0 <= nny < N:
                if graph[nnx][nny]:
                    count += 1
                    
        graph[nx][ny] += count
        
    for x in range(N):
        for y in range(N):
            if (x, y) not in not_cloud:
                if graph[x][y] >= 2:
                    clouds.append((x, y))
                    graph[x][y] -= 2

print(sum([sum(_) for _ in graph]))



# BOJ - 상어 초등학교(21608): G5
from collections import defaultdict

rx, ry = [0, 0, 1, -1], [1, -1, 0, 0]
N = int(input())
seats = [[0] * N for _ in range(N)]
students = [list(map(int, input().split())) for _ in range(N**2)]
friends = defaultdict(set)

for student in students:
    me = student[0]
    friends[me] = set(student[1:])
    possible = []
    
    for x in range(N):
        for y in range(N):
            if not seats[x][y]:
                empty = 0
                friend = 0
                
                for _ in range(4):
                    nx = rx[_] + x
                    ny = ry[_] + y
                    
                    if 0 <= nx < N and 0 <= ny < N:
                        if not seats[nx][ny]:
                            empty += 1
                        if seats[nx][ny] in friends[me]:
                            friend += 1
                            
                possible.append((friend, empty, x, y))
                
    possible.sort(key=lambda k: (-k[0], -k[1], k[2], k[3]))
    _, _, x, y = possible[0]
    seats[x][y] = me
    
answer = 0

for x in range(N):
    for y in range(N):
        me = seats[x][y]
        friend = 0
        
        for _ in range(4):
            nx = rx[_] + x
            ny = ry[_] + y
            
            if 0 <= nx < N and 0 <= ny < N:
                if seats[nx][ny] in friends[me]:
                    friend += 1
        if friend:
            answer += 10 ** (friend - 1)

print(answer)



# BOJ - 톱니바퀴(14891): G5
from collections import deque

def right(idx, d):
    if idx > 3:
        return

    if sawtooth[idx - 1][2] != sawtooth[idx][6]:
        right(idx + 1, -d)
        sawtooth[idx].rotate(d)

def left(idx, d):
    if idx < 0:
        return
    
    if sawtooth[idx][2] != sawtooth[idx + 1][6]:
        left(idx - 1, -d)
        sawtooth[idx].rotate(d)

sawtooth = [deque(list(map(int, input()))) for _ in range(4)]
k = int(input())

for _ in range(k):
    idx, d = map(int, input().split())
    idx -= 1
    left(idx-1, -d)
    right(idx+1, -d)

    sawtooth[idx].rotate(d)

score = 0

for i in range(4):
    if sawtooth[i][0] == 1:
        score += 2 ** i

print(score)



# BOJ - 뱀(3190): G4
from collections import deque

N = int(input())
K = int(input())
board = [[0] * N for _ in range(N)]

for _ in range(K):
    ax, ay = map(int, input().split())
    board[ax-1][ay-1] = 1

L = int(input())
turn = []

for _ in range(L):
    X, C = map(str, input().split())
    turn.append((int(X), C))
    
dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]
nd, hx, hy, t, i = 0, 0, 0, 0, 0
deq = deque()
deq.append((hx, hy))

while 1:
    hx = hx + dx[nd]
    hy = hy + dy[nd]
    t += 1

    if hx < 0 or hx >= N or hy < 0 or hy >= N or (hx, hy) in deq: break

    deq.append((hx, hy))

    if board[hx][hy] == 0:
        deq.popleft()
    else:
        board[hx][hy] = 0

    if t == turn[i][0]:
        if turn[i][1] == 'L':
            nd = (nd - 1) % 4
        else:
            nd = (nd + 1) % 4

        if i + 1 < len(turn): i += 1

print(t)



# 상담원 인원(214288) - Lv.3
from heapq import *
from itertools import *

def solution(k, n, reqs):
    answer = 10**9
    
    for com in combinations(range(1, n), k-1):
        com, heapq = [0, *com, n], [[] for _ in range(k+1)]
        S = 0
        
        for a, b, c in reqs:
            while heapq[c] and heapq[c][0] <= a:
                heappop(heapq[c])
                
            if len(heapq[c]) == com[c] - com[c-1]:
                d = heappop(heapq[c]) - a
                S += d
                b += d
                
            heappush(heapq[c], a+b)
            
        answer = min(answer, S)
        
    return answer



# 산 모양 타일링(258705) - Lv.3
def solution(n, tops):
    MOD = 10007
    dp1, dp2 = [0] * n, [0] * n
    dp1[0], dp2[0] = 1, 2 + tops[0]
    
    for i in range(1, n):
        dp1[i] = (dp1[i - 1] + dp2[i - 1]) % MOD
        dp2[i] = ((dp1[i - 1] * (1 + tops[i])) + (dp2[i - 1] * (2 + tops[i]))) % MOD
        
    return (dp1[-1] + dp2[-1]) % MOD



# 표현 가능한 이진트리(150367) - Lv.3
from collections import deque
 
def solution(numbers):
    answer = []
    
    for num in numbers:
        num = bin(num)[2:]
        n = 0
        
        for i in range(10000):
            if len(num) < 2 ** i:
                n = i
                break

        while len(num) != 2 ** n - 1:
            num = "0"+num

        root = len(num) // 2
        deq = deque([root])
        flag = True
        
        while n != 0 and flag:
            gap = 2 ** (n - 2)
            nextDeq = deque()
            
            if n == 1: break
            
            while deq:
                a = deq.popleft()
                left = a - gap
                right = a + gap
                
                if num[a] == "0":
                    if num[left] == "1" or num[right] == "1":
                        answer.append(0)
                        flag = False
                        break

                nextDeq.append(left)
                nextDeq.append(right)

            deq = nextDeq
            n -= 1

        if flag == True:
            answer.append(1)
            
    return answer



# 표 병합(150366) - Lv.3
def solution(commands):
    answer = []
    merged = [[(i, j) for j in range(50)] for i in range(50)]
    board = [["EMPTY"] * 50 for _ in range(50)]
    
    for command in commands:
        command = command.split(' ')
        
        if command[0] == 'UPDATE':
            if len(command) == 4:
                r, c, value = int(command[1]) - 1, int(command[2]) - 1,command[3]
                x, y = merged[r][c]
                board[x][y] = value
            elif len(command) == 3:
                value1, value2 = command[1], command[2]
                
                for i in range(50):
                    for j in range(50):
                        if board[i][j] == value1:
                            board[i][j] = value2
        elif command[0] == 'MERGE':
            r1, c1, r2, c2 = int(command[1]) - 1, int(command[2]) - 1, int(command[3]) - 1, int(command[4]) - 1
            x1, y1 = merged[r1][c1]
            x2, y2 = merged[r2][c2]
            
            if board[x1][y1] == "EMPTY":
                board[x1][y1] = board[x2][y2]
                
            for i in range(50):
                for j in range(50):
                    if merged[i][j] == (x2, y2):
                        merged[i][j] = (x1, y1)
        elif command[0] == 'UNMERGE':
            r, c = int(command[1]) - 1, int(command[2]) - 1
            x, y = merged[r][c]
            tmp = board[x][y]
            
            for i in range(50):
                for j in range(50):
                    if merged[i][j] == (x, y):
                        merged[i][j] = (i, j)
                        board[i][j] = "EMPTY"
                        
            board[r][c] = tmp
        elif command[0] == 'PRINT':
            r, c = int(command[1]) - 1, int(command[2]) - 1
            x, y = merged[r][c]
            answer.append(board[x][y])
            
    return answer



# 아이템 줍기(87694) - Lv.3
from collections import deque

def solution(rectangle, characterX, characterY, itemX, itemY):
    answer = 0
    graph = [[-1] * 102 for _ in range(102)]
    visited = [[0] * 102 for _ in range(102)]
    edges = set()
    
    for elem in rectangle:
        lx, ly, rx, ry = map(lambda x:x*2, elem)
        
        for i in range(lx, rx+1):
            for j in range(ly, ry+1):
                if lx < i < rx and ly < j < ry:
                    graph[i][j] = 0
                elif graph[i][j] != 0:
                    graph[i][j] = 1
            
    dxs, dys = [0,0,1,-1], [1,-1,0,0]
    q = deque()
    q.append((characterX*2, characterY*2))
    
    while q:
        x, y = q.popleft()
        
        if x == itemX * 2 and y == itemY * 2:
            answer = visited[x][y] // 2
            break
            
        for dx, dy in zip(dxs, dys):
            nx, ny = dx + x, dy + y
            
            if 0 < nx < 102 and 0 < ny < 102 and not visited[nx][ny]:
                if graph[nx][ny] == 1:
                    visited[nx][ny] = visited[x][y] + 1
                    q.append((nx,ny))
                    
    return answer



# 두 큐 합 같게 만들기(118667) - Lv.2
from collections import deque
 
def solution(queue1, queue2):
    answer = 0
    q1, q2 = deque(queue1), deque(queue2)
    sum1, sum2 = sum(q1), sum(q2)
    limit = len(q1) * 3
    
    while sum1 != sum2:
        if sum1 > sum2:
            sum1 -= q1[0]
            sum2 += q1[0]
            q2.append(q1.popleft())
        elif sum1 < sum2:
            sum1 += q2[0]
            sum2 -= q2[0]
            q1.append(q2.popleft())
            
        answer += 1
        
        if answer == limit: 
            return -1
        
    return answer



# 파괴되지 않은 건물(92344) - Lv.3
def solution(board, skill):
    N, M = len(board), len(board[0])
    prefixSum = [[0] * (M+1)  for _ in range(N+1)]
    
    for t, r1, c1, r2, c2, d in skill:
        v = -d if t == 1 else d
        prefixSum[r1][c1] += v
        prefixSum[r2+1][c2+1] += v
        prefixSum[r1][c2+1] -= v
        prefixSum[r2+1][c1] -= v
        
    for i in range(N+1):
        for j in range(M):
            prefixSum[i][j+1] += prefixSum[i][j]
            
    for j in range(M+1):
        for i in range(N):
            prefixSum[i+1][j] += prefixSum[i][j]
            
    return sum([1 for i in range(N) for j in range(M) if board[i][j] + prefixSum[i][j] > 0])



# 금과 은 운반하기(86053) - Lv.3
def solution(a, b, g, s, w, t):
    left, right = 0, 10e15
    result = right 
    
    while left <= right:
        mid = (left + right) // 2
        gold, silver, total = 0, 0, 0

        for i in range(len(t)):
            cnt = (mid // (t[i] * 2))
            cnt += 1 if mid % (t[i] * 2) >= t[i] else 0
            gold += min(g[i], cnt * w[i])
            silver += min(s[i], cnt * w[i])
            total += min(g[i] + s[i], cnt * w[i])

        if gold >= a and silver >= b and total >= a + b:
            right = mid - 1
            result = min(mid, result)
        else:
            left = mid + 1

    return result



# 거리두기 확인하기(81302) - Lv.2
def check_distance(place):
    plist = [(y, x) for y in range(5) for x in range(5) if place[y][x] == 'P']
    
    for y, x in plist:
        for y2, x2 in plist:
            dist = abs(y - y2) + abs(x - x2)
            
            if dist == 0 or dist > 2:
                continue
            
            if dist == 1:
                return 0
            elif y == y2 and place[y][int((x+x2)/2)] != 'X':
                return 0
            elif x == x2 and place[int((y+y2)/2)][x] != 'X':
                return 0
            elif y != y2 and x != x2:
                if place[y2][x] != 'X' or place[y][x2] != 'X':
                    return 0
                
    return 1


def solution(places):
    answer = []
    
    for place in places:
        answer.append(check_distance(place))

    return answer



# [PCCP 기출문제] 2번 / 퍼즐 게임 챌린지(340212) - Lv.2
def solution(diffs, times, limit):
    left, right = 1, max(diffs)
    answer = right

    def solveTime(level):
        total = 0
        prev = times[0]

        for i in range(len(diffs)):
            diff = diffs[i]
            cur = times[i]

            if diff <= level: total += cur
            else:
                mistakes = diff - level
                total += (mistakes * (cur + prev)) + cur
            prev = cur

        return total

    while left <= right:
        mid = (left + right) // 2
        if solveTime(mid) <= limit:
            answer = mid
            right = mid - 1
        else: left = mid + 1

    return answer



# 삼각 달팽이(68645) - Lv.2
def solution(n):    
    answer = [[0 for j in range(1, i+1)] for i in range(1, n+1)]   
    x, y = -1, 0
    num = 1     
    
    for i in range(n): 
        for j in range(i, n):
            if i % 3 == 0:
                x += 1            
            elif i % 3 == 1:
                y += 1 
            else:
                x -= 1
                y -= 1 
                
            answer[x][y] = num
            num += 1     
            
    return sum(answer, [])



# 마법의 엘리베이터(148653) - Lv.2
def solution(storey):
    if storey < 10 :
        return min(storey, 11 - storey)
    
    left = storey % 10
    
    return min(left + solution(storey // 10), 10 - left + solution(storey // 10 + 1))



# 다단계 칫솔 판매(77486) - Lv.3
def solution(enroll, referral, seller, amount):
    money = [0 for _ in range(len(enroll))]
    dict = {}
    
    for i, e in enumerate(enroll):
        dict[e] = i
        
    for s, a in zip(seller, amount):
        m = a * 100
        
        while s != "-" and m > 0:
            idx = dict[s]
            money[idx] += m - m//10
            m //= 10
            s = referral[idx]
            
    return money



# 보석 쇼핑(67258) - Lv.3
def solution(gems):
    N = len(gems)
    answer = [0, N-1]
    kind = len(set(gems))
    dic = { gems[0]: 1, }
    s, e = 0, 0 
    
    while s < N and e < N:
        if len(dic) < kind:  
            e += 1
            
            if e == N: break
                
            dic[gems[e]] = dic.get(gems[e], 0) + 1
        else: 
            if (e - s + 1) < (answer[1] - answer[0] + 1): answer = [s, e]
                
            if dic[gems[s]] == 1:
                del dic[gems[s]]
            else:
                dic[gems[s]] -= 1
                
            s += 1

    answer[0] += 1
    answer[1] += 1
    
    return answer



# 괄호 회전하기(76502) - Lv.2
def solution(s):
    answer = 0
    s = list(s)
    
    for _ in range(len(s)):
        stack = []
        
        for i in range(len(s)):
            if len(stack) > 0:
                if stack[-1] == '[' and s[i] == ']': stack.pop()
                elif stack[-1] == '{' and s[i] == '}': stack.pop()
                elif stack[-1] == '(' and s[i] == ')': stack.pop()
                else: stack.append(s[i])
            else:
                stack.append(s[i])
                
        if len(stack) == 0:
            answer += 1
            
        s.append(s.pop(0))

    return answer



# 카드 짝 맞추기(72415) - Lv.3
from collections import deque
from itertools import permutations
from copy import deepcopy

board = []

def ctrl_move(r, c, k, t):
    global board
    cr, cc = r, c
    
    while True:
        nr = cr + k
        nc = cc + t
        
        if not (0 <= nr < 4 and 0 <= nc < 4):
            return cr, cc
        
        if board[nr][nc] != 0:
            return nr, nc
    
        cr = nr
        cc = nc

        
def bfs(start, end):
    r, c = start
    find_r, find_c = end
    queue = deque()
    queue.append((r, c, 0))
    visited = [[0]*4 for _ in range(4)]
    move = [(0,-1),(0,1),(-1,0),(1,0)]
    
    while queue:
        r, c, temp = queue.popleft()
        
        if visited[r][c]:
            continue
        
        visited[r][c] = 1
        
        if r == find_r and c == find_c:
            return temp
    
        for k, t in move:
            cr = k + r
            cc = c + t
            
            if 0 <= cr < 4 and 0 <= cc < 4:
                queue.append((cr,cc, temp+1))
            
            cr, cc = ctrl_move(r, c, k, t)
            queue.append((cr, cc, temp+1))
            
    return -1


def solution(input_board, sr, sc):
    global board
    board = input_board
    location = [[] for _ in range(7)]
    nums = []
    
    for i in range(4):
        for j in range(4):
            if board[i][j]:
                if board[i][j] not in nums:
                    nums.append(board[i][j])
                    
                location[board[i][j]].append((i, j))
                
    per = list(permutations(nums, len(nums)))
    answer = float('inf')
    
    for i in range(len(per)):
        board = deepcopy(input_board)
        cnt = 0
        r, c = sr, sc
        
        for j in per[i]:
            left = bfs((r, c), location[j][0])
            right = bfs((r, c), location[j][1])
            
            if left < right:
                cnt += left
                cnt += bfs(location[j][0], location[j][1])
                r, c = location[j][1]
            else:
                cnt += right
                cnt += bfs(location[j][1], location[j][0])
                r, c = location[j][0]
                
            board[location[j][0][0]][location[j][0][1]] = 0
            board[location[j][1][0]][location[j][1][1]] = 0
            cnt += 2 
            
        answer = min(answer, cnt)
        
    return answer



# 외벽 점검(60062) - Lv.3
from collections import deque

def solution(n, weak, dist):
    dist.sort(reverse=True)
    q = deque([weak])
    visited = set()
    visited.add(tuple(weak))
    
    for i in range(len(dist)):
        d = dist[i]
        
        for _ in range(len(q)):
            current = q.popleft()
            
            for p in current:
                l, r = p, (p + d) % n
                
                if l < r:
                    temp = tuple(filter(lambda x: x < l or x > r, current))
                else:
                    temp = tuple(filter(lambda x: x < l and x > r, current))

                if len(temp) == 0:
                    return (i + 1)
                elif temp not in visited:
                    visited.add(temp)
                    q.append(list(temp))
                    
    return -1



# 행렬 테두리 회전하기(77485) - Lv.2
def solution(rows, columns, queries):
    arr = [[0 for c in range(columns)] for r in range(rows)]
    answer = []
    t = 1
    
    for row in range(rows):
        for col in range(columns):
            arr[row][col] = t
            t += 1

    for x1, y1, x2, y2 in queries:
        tmp = arr[x1-1][y1-1]
        mini = tmp

        for k in range(x1-1,x2-1):
            test = arr[k+1][y1-1]
            arr[k][y1-1] = test
            mini = min(mini, test)

        for k in range(y1-1,y2-1):
            test = arr[x2-1][k+1]
            arr[x2-1][k] = test
            mini = min(mini, test)

        for k in range(x2-1,x1-1,-1):
            test = arr[k-1][y2-1]
            arr[k][y2-1] = test
            mini = min(mini, test)

        for k in range(y2-1,y1-1,-1):
            test = arr[x1-1][k-1]
            arr[x1-1][k] = test
            mini = min(mini, test)

        arr[x1-1][y1] = tmp
        answer.append(mini)
        
    return answer



# 모음 사전(84512) - Lv.2
def solution(word):
    answer = 0
    
    for i, n in enumerate(word):
        answer += (5 ** (5 - i) - 1) / (5 - 1) * "AEIOU".index(n) + 1  # 등비수열의 합
        
    return answer



# 멀쩡한 사각형(62048) - Lv.2
import math

def solution(w,h):
    return w * h - (w + h - math.gcd(w, h))



# 불량 사용자(64064) - Lv.3
from itertools import product

def solution(user_id, banned_id):
    banned, result = [], []
    
    for ban in banned_id:
        temp = []
        
        for user in user_id:
            if len(user) == len(ban):
                flag = True
                
                for i in range(len(ban)):
                    if ban[i] != user[i] and ban[i] != '*':
                        flag = False
                        break
                        
                if flag: temp.append(user)
                
        banned.append(temp)
        
    for p in product(*banned):
        if len(set(p)) == len(banned_id):
            flag = True
            
            for s in result:
                if len(s - set(p)) == 0:
                    flag = False
                    break
                    
            if flag: result.append(set(p))
    
    return len(result)



# 유사 칸토어 비트열(148652) - Lv.2
def solution(n, l, r):
    answer = r - l + 1
    
    for num in range(l-1, r):
        while num >= 1:
            a, b = divmod(num, 5)
            
            if b == 2 or a == 2:
                answer -= 1
                break
                
            num = a

    return answer



# 할인 행사(131127) - Lv.2
from collections import Counter

def solution(want, number, discount):
    d = 0
    check = {}
    
    for w, n in zip(want, number):
        check[w] = n
    
    for i in range(len(discount) - 9):
        c = Counter(discount[i:i+10])
        
        if c == check: d += 1

    return d



# 교점에 별 만들기(87377) - Lv.2
def solution(line):
    answer = []    
    points = set()
    
    for i in range(len(line)):
        for j in range(i + 1, len(line)):
            a, b, e = line[i]
            c, d, f = line[j]
            
            if (a * d) - (b * c) != 0:
                x = (b * f - e * d) / (a * d - b * c)
                y = (e * c - a * f) / (a * d - b * c)
            
            if int(x) == x and int(y) == y:
                x = int(x)
                y = int(y)
                points.add((x, y))
                
    minX = min(point[0] for point in points)
    minY = min(point[1] for point in points)
    maxX = max(point[0] for point in points)
    maxY = max(point[1] for point in points)
    
    for y in range(maxY, minY - 1, -1):
        row = ""
        
        for x in range(minX, maxX + 1):
            if (x, y) in points:
                row += "*"
            else:
                row += "."
                
        answer.append(row)
                    
    return answer



# 전력망을 둘로 나누기(86971) - Lv.2
def dfs(v, graph, visited):
    visited[v] = True

    return sum([1] + [dfs(u, graph, visited) for u in graph[v] if not visited[u]])


def solution(n, wires):
    graph = [[] for _ in range(n+1)]

    for v, u in wires:
        graph[v].append(u)
        graph[u].append(v)

    answer = 100

    for i in range(n-1):
        visited = [False for _ in range(n+1)]
        v1, v2 = wires[i]
        visited[v2] = True
        tmp = abs(dfs(v1, graph, visited) - dfs(v2, graph, visited))
        answer = min(tmp, answer)

    return answer



# 숫자 블록(12923) - Lv.2
def get_div(n):
    if n == 1:
        return 0
    elif n <= 3:
        return 1

    tmp = []
    
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            d = int(n / i)
            
            if d <= 10000000:
                return d
            else:
                tmp.append(i)
                
    if tmp: return tmp[-1]
    
    return 1

def solution(begin, end):
    answer = [get_div(i) for i in range(begin, end+1)]
    
    return answer



# 메뉴 리뉴얼(72411) - Lv.2
from itertools import combinations
from collections import Counter

def solution(orders, course):
    answer = []
    
    for k in course:
        candidates = []
        
        for menu_li in orders:
            for li in combinations(menu_li, k):
                res = ''.join(sorted(li))
                candidates.append(res)
                
        sorted_candidates = Counter(candidates).most_common()
        answer += [menu for menu, cnt in sorted_candidates if cnt > 1 and cnt == sorted_candidates[0][1]]
        
    return sorted(answer)



# 광고 삽입(72414) - Lv.3
def solution(play, adv, logs):
    c = lambda t: int(t[0:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8])
    play, adv = c(play), c(adv)
    logs = sorted([s for t in logs for s in [(c(t[:8]), 1), (c(t[9:]), 0)]])

    v, p, b = 0, 0, [0] * play

    for t, m in logs:
        if v > 0: b[p:t] = [v] * (t - p)

        v, p = v + (1 if m else -1), t

    mv, mi = (s := sum(b[:adv]), 0)

    for i, j in zip(range(play - adv), range(adv, play)):
        s += b[j] - b[i]

        if s > mv: mv, mi = s, i + 1

    return f"{mi//3600:02d}:{mi%3600//60:02d}:{mi%60:02d}"



# BOJ - 로봇 청소기(14503): G5
from collections import deque

N, M = map(int, input().split())
r, c, d = map(int, input().split())
room = [list(map(int, input().split())) for _ in range(N)]
visited = [[False for _ in range(M)] for _ in range(N)]
dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]

deq = deque()
deq.append((r, c))
visited[r][c] = True
cnt = 1

while deq:
    x, y = deq.popleft()
    flag = 0

    for _ in range(4):
        d = (d + 3) % 4
        nx = x + dx[d]
        ny = y + dy[d]

        if 0 <= nx < N and 0 <= ny < M and not room[nx][ny]: 
            if not visited[nx][ny]:
                visited[nx][ny] = 1  
                deq.append((nx, ny))
                cnt += 1
                flag = 1
                break

    if flag == 0:
        if room[x-dx[d]][y-dy[d]] != 1:
            deq.append((x-dx[d], y-dy[d]))
        else:
            print(cnt)
            break



# BOJ - 컨베이어 벨트 위의 로봇(20055): G5
from collections import deque

N, K = map(int,input().split())
A = deque(list(map(int, input().split())))
answer = 0
belt = deque([False] * N)

while True:
    answer += 1
    A.rotate(1)
    belt.rotate(1)
    belt[N-1] = False

    for i in range(N-2, -1, -1):
        if belt[i] and not belt[i+1] and A[i+1] > 0:
            belt[i], belt[i+1] = False, True
            A[i+1] -= 1
    
    belt[N-1] = False
        
    if A[0] > 0:
        belt[0] = True
        A[0] -= 1
        
    if A.count(0) >= K: break

print(answer)



# BOJ - 테트로미노(14500): G4
N, M = map(int, input().split())
paper = [list(map(int, input().split())) for _ in range(N)]
visited = [([False] * M) for _ in range(N)]
dr, dc = [-1, 0, 1, 0], [0, 1, 0, -1]
MAX = max(map(max, paper))
sum = 0

def dfs(r, c, l, t):
    global sum
    
    if sum >= t + MAX * (3-l): return
    
    if l == 3:
        sum = max(sum, t)
        return
    else:
        for i in range(4):
            nr = r + dr[i]
            nc = c + dc[i]
            
            if 0 <= nr < N and 0 <= nc < M and visited[nr][nc] == False:
                if l == 1:
                    visited[nr][nc] = True
                    dfs(r, c, l+1, t+paper[nr][nc])
                    visited[nr][nc] = False
                visited[nr][nc] = True
                dfs(nr, nc, l+1, t+paper[nr][nc])
                visited[nr][nc] = False

for r in range(N):
    for c in range(M):
        visited[r][c] = True
        dfs(r, c, 0, paper[r][c])
        visited[r][c] = False

print(sum)



# 110 옮기기(77886) - Lv.3
def f(x):
    left, I, IIO = '', 0, 0

    for i in x:
        if i == '1':
            I += 1
        elif I > 1:
            I -= 2
            IIO += 1
        else:
            left += '10' if I > 0 else '0'
            I = 0
            
    return left + '110' * IIO + '1' * I

def solution(s):
    return [f(x) for x in s]



# BOJ - 인구 이동(16234): G4
from collections import deque

N, L, R = map(int, input().split())
board = []
dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]
day = 0

def bfs(i, j):
    group = []
    q = deque()
    q.append([i, j])
    group.append([i, j])

    while q:
        x, y = q.popleft()

        for k in range(4):
            nx, ny = x+dx[k], y+dy[k]

            if 0 <= nx < N and 0 <= ny < N and visited[nx][ny] == 0:
                if L <= abs(board[nx][ny]-board[x][y]) <= R:
                    group.append([nx, ny])
                    q.append([nx, ny])
                    visited[nx][ny] = 1

    return group

for _ in range(N):
    board.append(list(map(int, input().split())))

while True:
    visited = [[0] * N for _ in range(N)]
    finalflag = False

    for i in range(N):
        for j in range(N):
            if visited[i][j] == 0:
                visited[i][j] = 1
                
                group = bfs(i, j)

                if len(group) > 1:
                    finalflag = True
                    change_value = sum([board[x][y] for x, y in group]) // len(group)
                    
                    for x, y in group: board[x][y] = change_value
                        
    if not finalflag: break
        
    day+=1

print(day)



# 숫자 카드 나누기(135807) - Lv.2
from math import gcd

def solution(arrayA, arrayB):
    a = arrayA[0]
    b = arrayB[0]

    for i in range(len(arrayA)):
        a = gcd(a, arrayA[i])
        b = gcd(b, arrayB[i])
        
    cA, cB = 1, 1
    
    for i in range(len(arrayA)):
        if arrayA[i] % b == 0: cA = 0
        if arrayB[i] % a == 0: cB = 0

    return cA if cA == 0 and cB == 0 else max(a, b)



# 양과 늑대(92343) - Lv.3
def solution(info, edges):
    answer = []
    visited = [False] * len(info)
    
    def dfs(sheeps, wolves):
        if sheeps > wolves: answer.append(sheeps)
        else: return
        
        for p, c in edges:
            if visited[p] and not visited[c]:
                visited[c] = True
                
                if info[c] == 0: dfs(sheeps + 1, wolves)
                else: dfs(sheeps, wolves + 1)
                
                visited[c] = False
                
    visited[0] = True
    dfs(1, 0)
    
    return max(answer)



# 합승 택시 요금(72413) - Lv.3
from itertools import product

def solution(n, s, a, b, fares):
    s, a, b = s-1, a-1, b-1
    d = [[100000000] * n for _ in range(n)]

    for t, f, v in fares: 
        d[t - 1][f - 1] = d[f - 1][t - 1] = v

    for t in range(n): 
        d[t][t] = 0

    for k, i, j in product(range(n), repeat=3):
        if d[i][j] > (l := d[i][k] + d[k][j]): d[i][j] = l

    return min([d[s][k]+d[k][a]+d[k][b] for k in range(n)])



# 기둥과 보 설치(60061) - Lv.3
def check(answer):
    for x, y, stuff in answer:
        if stuff == 0:
            if y == 0 or [x-1, y, 1] in answer or [x, y, 1] in answer or [x, y-1, 0] in answer: continue
            return False
        elif stuff == 1: 
            if [x, y-1, 0] in answer or [x+1, y-1, 0] in answer or ([x-1, y, 1] in answer and [x+1, y, 1] in answer): continue
            return False
        
    return True

def solution(n, build_frame):
    answer = []
    
    for build in build_frame:
        x, y, stuff, operation = build
        
        if operation == 1:
            answer.append([x, y, stuff])
            if not check(answer): answer.remove([x, y, stuff])
        elif operation == 0:
            answer.remove([x, y, stuff])
            if not check(answer): answer.append([x, y, stuff])
            
    answer.sort()
    
    return answer



# 길 찾기 게임(42892) - Lv.3
import sys; sys.setrecursionlimit(10001)

pre, post = list(), list()

def solution(nodeinfo):
    l = sorted(list({x[1] for x in nodeinfo}), reverse=True)
    n = sorted(list(zip(range(1, len(nodeinfo)+1), nodeinfo)), key=lambda x:(-x[1][1], x[1][0]))
    order(n, l, 0)
    
    return [pre, post]

def order(nodes, levels, curlevel):
    n = nodes[:]
    cur = n.pop(0)
    pre.append(cur[0])
    
    if n: 
        for i in range(len(n)):
            if n[i][1][1] == levels[curlevel+1]:
                if n[i][1][0] < cur[1][0]:
                    order([x for x in n if x[1][0] < cur[1][0]], levels, curlevel+1)
                else:
                    order([x for x in n if x[1][0] > cur[1][0]], levels, curlevel+1)
                    break

    post.append(cur[0])



# [1차] 뉴스 클러스터링(17677) - Lv.3
import re
import math

def solution(str1, str2):
    str1 = [str1[i:i+2].lower() for i in range(0, len(str1)-1) if not re.findall('[^a-zA-Z]+', str1[i:i+2])]
    str2 = [str2[i:i+2].lower() for i in range(0, len(str2)-1) if not re.findall('[^a-zA-Z]+', str2[i:i+2])]

    gyo = set(str1) & set(str2)
    hap = set(str1) | set(str2)

    if len(hap) == 0: return 65536

    gyo_sum = sum([min(str1.count(gg), str2.count(gg)) for gg in gyo])
    hap_sum = sum([max(str1.count(hh), str2.count(hh)) for hh in hap])

    return math.floor((gyo_sum/hap_sum)*65536)



# 스티커 모으기(2)(12971) - Lv.3
def solution(sticker):
    size = len(sticker)
    
    if len(sticker) == 1: return sticker.pop()

    dp1 = [0] + sticker[:-1]
    
    for i in range(2, size):
        dp1[i] = max(dp1[i-1], dp1[i-2] + dp1[i])
    
    dp2 = [0] + sticker[1:]
    
    for i in range(2, size):
        dp2[i] = max(dp2[i-1], dp2[i-2] + dp2[i])
    
    return max(dp1[-1], dp2[-1])



# [3차] 방금그곡(17683) - Lv.2
def solution(m, musicinfos):
    answer = ''
    result = list()
    
    for i in musicinfos:
        info = i.split(',')
        hour = int(info[1].split(':')[0]) - int(info[0].split(':')[0])
        minute = int(info[1].split(':')[1]) - int(info[0].split(':')[1])
        time = hour * 60 + minute
        music = list(info[-1])
        i = 1
        
        while (i < len(music)) and ('#' in music):
            if music[i] == '#':
                music[i - 1] = music[i - 1] + music[i]
                del music[i]
            else:
                i += 1

        length = len(music)
        play = music * (time // length) + music[:time % length]
        
        m = list(m)
        i = 1
        
        while (i < len(m)) and ('#' in m):
            if m[i] == '#':
                m[i - 1] = m[i - 1] + m[i]
                del m[i]
            else:
                i += 1

        for i in range(len(play) - len(m) + 1):
            if play[i:i+len(m)] == m:
                result.append([info[2], time])
                break
        
    if len(result) > 0:
        result.sort(key = lambda x : x[1], reverse = True)
        answer = result[0][0]
    else:
        answer = '(None)'
    
    return answer



# BOJ - 연구소(14502): G4
from collections import deque
from copy import deepcopy
from itertools import combinations

N, M = map(int, input().split())
graph = [list(map(int, input().split())) for _ in range(N)]
point = [(x, y) for x in range(M) for y in range(N) if not graph[y][x]]
answer = 0

def bfs(graph):
    global answer
    deq = deque([(j, i) for i in range(N) for j in range(M) if graph[i][j] == 2])

    while deq:
        x, y = deq.popleft()

        for nx, ny in zip([x+1, x-1, x, x], [y, y, y+1, y-1]):
            if 0 <= nx < M and 0 <= ny < N and not graph[ny][nx]:
                graph[ny][nx] = 2
                deq.append((nx, ny))
    
    count = sum([i.count(0) for i in graph])
    answer = max(answer, count)

for c in combinations(point, 3):
    tmp = deepcopy(graph)

    for x, y in c: 
        tmp[y][x] = 1

    bfs(tmp)

print(answer)



# 경주로 건설(67259) - Lv.3
from heapq import heappop, heappush
from sys import maxsize

def solution(board):
    N = len(board)
    costBoard = [[[maxsize] * N for _ in range(N)] for _ in range(4)]
    heap = [(0, 0, 0, 0), (0, 0, 0, 2)]
    
    for i in range(4): 
        costBoard[i][0][0] = 0
    
    while heap:
        cost, x, y, d = heappop(heap)
        
        for dx, dy, dd in ((1, 0, 0), (-1, 0, 1), (0, 1, 2), (0, -1, 3)):
            nx, ny = x + dx, y + dy
            
            if nx < 0 or nx >= N or ny < 0 or ny >= N or board[ny][nx]: continue
            
            newCost = cost + (100 if d == dd else 600)
            
            if costBoard[dd][ny][nx] > newCost:
                costBoard[dd][ny][nx] = newCost
                heappush(heap, (newCost,nx,ny,dd))
                
    return min(costBoard[0][N-1][N-1], costBoard[1][N-1][N-1], costBoard[2][N-1][N-1], costBoard[3][N-1][N-1])



# 부대복귀(132266) - Lv.3
from collections import deque

def solution(n, roads, sources, destination):
    path = [[] for _ in range(n+1)]
    visited = [-1] * (n+1)
    
    for a, b in roads:
        path[a].append(b)
        path[b].append(a)

    deq = deque([destination])
    visited[destination] = 0
    
    while deq:
        now = deq.popleft()

        for n in path[now]:
            if visited[n] == -1:
                visited[n] = visited[now] + 1
                deq.append(n)

    return [visited[i] for i in sources]



# k진수에서 소수 개수 구하기(92335) - Lv.2
def solution(n, k):
    word = ""
    
    while n:
        word = str(n % k) + word
        n = n // k
        
    word = word.split("0")
    count = 0
    
    for w in word:
        if len(w) == 0: continue
        if int(w) < 2: continue
        
        prime = True
        
        for i in range(2, int(int(w) ** 0.5) + 1):
            if int(w) % i == 0:
                prime = False
                break
                
        if prime: count += 1
            
    return count



# 다리를 지나는 트럭(42583) - Lv.2
from collections import deque

def solution(bridge_length, weight, truck_weights):
    time, current = 0, 0
    bridge = deque([0] * bridge_length)
    truck_weights = deque(truck_weights)
    
    while len(truck_weights) > 0:
        time = time + 1
        current = current - bridge.popleft()

        if current + truck_weights[0] <= weight:
            current = current + truck_weights[0]
            bridge.append(truck_weights.popleft())
        else: 
            bridge.append(0)
            
    time = time + bridge_length
    
    return time



# 조이스틱(42860) - Lv.2
def solution(name):
    answer = 0
    least = len(name) - 1
    
    for i, c in enumerate(name):
        answer += min(ord(c) - ord('A'), ord('Z') - ord(c) + 1)
        next = i + 1
        
        while next < len(name) and name[next] == 'A':
            next += 1
        
        least = min([least, 2*i+len(name)-next, i+2*(len(name)-next)])
        
    answer += least
    
    return answer



# 구명보트(42885) - Lv.2
def solution(people, limit) :
    answer = 0
    a, b = 0, len(people) - 1
    people.sort()
    
    while a < b:
        if people[b] + people[a] <= limit :
            a += 1
            answer += 1
            
        b -= 1
        
    return len(people) - answer



# [3차] n진수 게임(17687) - Lv.2
def solution(n, t, m, p):
    data = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F"]
    numbers = "0"
    
    for num in range(1, t*m):
        temp = ''
        
        while num > 0:
            temp = data[num % n] + temp
            num //= n
            
        numbers += temp

    return numbers[p-1:t*m:m]



# [1차] 캐시(17680) - Lv.2
from collections import deque

def solution(cacheSize, cities):
    cache = deque(maxlen=cacheSize)
    time = 0
    
    for i in cities:
        s = i.lower()
        
        if s in cache:
            cache.remove(s)
            cache.append(s)
            time += 1
        else:
            cache.append(s)
            time += 5
            
    return time



# [1차] 추석 트래픽(17676) - Lv.3
def throughput(log, start, end):
    cnt = 0
    
    for x in log:
        if x[0] < end and x[1] >= start: cnt += 1
        
    return cnt

def solution(lines):
    answer = 0
    log = []

    for line in lines:
        date, s, t = line.split()
        s, t = s.split(':'), t.replace('s', '')
        end = (int(s[0])*3600 + int(s[1])*60 + float(s[2]))*1000 
        start = end - float(t) * 1000 + 1
        log.append([start, end])

    for x in log:
        answer = max(answer, throughput(log, x[0], x[0] + 1000), throughput(log, x[1], x[1] + 1000))

    return answer



# [3차] 압축(17684) - Lv.2
def solution(msg):
    answer = []
    dic = {chr(i+64): i for i in range(1, 27)}
    w = c = 0
    
    while True:
        c += 1	
        
        if c == len(msg):	
            answer.append((dic[msg[w:c]]))
            break
        
        if msg[w:c+1] not in dic:
            dic[msg[w:c+1]] = len(dic) + 1
            answer.append(dic[msg[w:c]])
            w = c
            
    return answer



# 줄 서는 방법(12936) - Lv.2
from math import factorial

def solution(n, k):
    answer = []
    number = [i for i in range(1, n + 1)]

    while number:
        perm = factorial(n - 1)
        idx = (k - 1) // perm 
        answer.append(number.pop(idx))
        k %= perm 
        n -= 1

    return answer



# 최적의 행렬 곱셈(12942) - Lv.3
def solution(matrix_sizes):
    dp = [[0] * L for _ in range(L)]
    L = len(matrix_sizes)
    size = [s for s, _ in matrix_sizes]
    size.append(matrix_sizes[-1][-1])
    
    for i in range(1, L):
        for j in range(i):
            x, y = i-j-1, i
            dp[x][y] = min(size[x] * size[k+1] * size[y+1] + dp[x][k] + dp[k+1][y] for k in range(x, y))
            
    return dp[0][-1]



# H-Index(42747) - Lv.2
def solution(citations):
    citations.sort(reverse=True)
    answer = max(map(min, enumerate(citations, start=1)))
    
    return answer



# 야근 지수(12927) - Lv.3
import heapq

def solution(n, works):
    if n >= sum(works): return 0
    
    works = [-w for w in works]
    heapq.heapify(works)
    
    for _ in range(n):
        i = heapq.heappop(works)
        i += 1
        heapq.heappush(works, i)
    
    return sum([w ** 2 for w in works])



# 표 편집(81303) - Lv.3
def solution(n, k, cmd):
    cur = k 
    deleted = [] 
    table = {i: [i - 1, i + 1] for i in range(n)}
    answer = ['O'] * n 
    
    for c in cmd:
        if c[0] == "C": 
            deleted.append([cur, answer[cur]])  
            answer[cur] = 'X'  
            prev, nxt = table[cur] 
            
            if prev != -1: table[prev][1] = nxt 
            if nxt != n: table[nxt][0] = prev 
            
            if nxt == n:
                cur = prev 
            else:
                cur = nxt 
        elif c[0] == "Z":
            restore, status = deleted.pop()
            answer[restore] = status
            prev, nxt = table[restore] 
            
            if prev != -1: table[prev][1] = restore
            if nxt != n: table[nxt][0] = restore
        else:
            direction, count = c.split(' ')
            count = int(count)
            
            if direction == 'D':  
                for _ in range(count):
                    cur = table[cur][1]
            else: 
                for _ in range(count):
                    cur = table[cur][0]

    return ''.join(answer) 



# 네트워크(43162) - Lv.3
def DFS(n, computers, c, visited):
    visited[c] = True 
    
    for connect in range(n):
        if connect != c and computers[c][connect] == 1 and visited[connect] == False: 
            DFS(n, computers, connect, visited)

def solution(n, computers):  
    answer = 0
    visited = [False for _ in range(n)]
    
    for c in range(n):                        
        if visited[c] == False:            
            DFS(n, computers, c, visited) 
            answer += 1                     
            
    return answer



# 등굣길(42898) - Lv.3
def solution(m, n, puddles):
    puddles = [[q,p] for [p,q] in puddles] 
    dp = [[0] * (m+1) for i in range(n+1)]
    dp[1][1] = 1 

    for i in range(1, n+1):
        for j in range(1, m+1):
            if i == 1 and j == 1: continue 
            
            if [i, j] in puddles:   
                dp[i][j] = 0
            else:                    
                dp[i][j] = (dp[i-1][j] + dp[i][j-1]) % 1000000007
                
    return dp[n][m]



# 가장 긴 팰린드롬(12904) - Lv.3
def solution(s):
    answer = 0
    
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            if s[i:j] == s[i:j][::-1]: answer = max(answer, len(s[i:j]))
            
    return answer



# 카운트 다운(131129) - Lv.3
def solution(target):
    max_score = 100001
    possible_score = [[50, 1]]
    dp = [[] for _ in range(max(target + 1, 61))]
    
    for i in range(1, 21):
        possible_score.append([i, 1])
        if i * 2 > 20: possible_score.append([i * 2, 0])
        if i * 3 > 20 and ((i * 3) % 2 != 0 or i * 3 > 40): possible_score.append([i * 3, 0])
    
    for i in range(1, 21):
        dp[i] = [1, 1]
        dp[i * 2] = [1, 0]
        dp[i * 3] = [1, 0]
        
    dp[50] = [1, 1]
    
    for i in range(1, target + 1):
        temp_arr = []
        
        if len(dp[i]) == 0:
            for p, is_sb in possible_score:
                if i - p > 0: temp_arr.append([dp[i - p][0] + 1, dp[i - p][1] + is_sb])

            temp_arr = sorted(temp_arr, key = lambda x : (x[0], -x[1]))
            dp[i] = temp_arr[0]
            
    return dp[target]



# 스킬트리(49993) - Lv.2
def solution(skill, skill_trees):
    answer = 0

    for tree in skill_trees:
        s = ""
        
        for character in tree:
            if character in skill:
                s += character

        if skill[:len(s)] == s: answer += 1

    return answer



# 등대(133500) - Lv.3
from collections import defaultdict, deque

def solution(n, lighthouse):
    graph = defaultdict(list)
    onoff = [0 for _ in range(n + 1)]
    q = deque()

    for a, b in lighthouse:
        graph[a].append(b)
        graph[b].append(a)

    
    for i in range(1, n+1):
        if len(graph[i]) == 1: q.append(i)

    while q:
        now_leaf = q.popleft()
        if graph[now_leaf] == []: break
        parent = graph[now_leaf][0]

        del graph[now_leaf]
        graph[parent].remove(now_leaf)
        
        if len(graph[parent]) == 1: q.append(parent)
        if onoff[now_leaf] == 1: continue
        onoff[parent] = 1
    
    return sum(onoff) 



# 카펫(42842) - Lv.2
def solution(brown, yellow):    
    for bh in range(1, brown//2 + 1):
        bw = (brown - 2 * bh + 4) // 2
        yw, yh = bw - 2, bh - 2
        
        if yellow == yw * yh and yellow + brown == bw * bh: return [bw, bh]



# 미로 탈출 명령어(150365) - Lv.3
def solution(n, m, x, y, r, c, k):
    answer = ''
    diff = abs(x-r) + abs(y-c)
    
    if diff%2 != k%2 or diff > k: return 'impossible'

    rest = k-diff
    lc, rc, dc, uc = 0, 0, 0, 0
    
    if x < r: dc = r - x
    else: uc = x - r
    
    if y < c: rc = c - y
    else: lc = y - c

    dplus = min(n-max(x, r), rest // 2)
    rest -= dplus * 2
    lplus = min(min(y, c)-1, rest // 2)
    rest -= lplus * 2
    answer = 'd' * (dc+dplus) + 'l' * (lc+lplus) + 'rl' * (rest//2) + 'r' * (rc+lplus) + 'u'*(dplus+uc)

    return answer



# 최고의 집합(12938) - Lv.3
def solution(n, s):
    answer = []
    
    if n > s:
        answer = [-1]
    else:
        for _ in range(n): answer.append(s // n)

        for i in range(s % n): answer[i] += 1

        answer.sort()
        
    return answer



# 거스름돈(12907) - Lv.3
def solution(n, money):
    dp = [0 for i in range(n + 1)]
    dp[0] = 1
    
    for c in money:
        for p in range(c, n+1):
            if p >= c: dp[p] += dp[p-c]

    return dp[n]



# 프로세스(42587) - Lv.2
def solution(priorities, location):
    queue =  [(i,p) for i, p in enumerate(priorities)]
    answer = 0
    
    while True:
        cur = queue.pop(0)
        
        if any(cur[1] < q[1] for q in queue): 
            queue.append(cur)
        else:
            answer += 1
            if cur[0] == location: return answer



# 더 맵게(42626) - Lv.2
import heapq as hq

def solution(scoville, K):
    cnt = 0
    hq.heapify(scoville)
    
    while len(scoville) > 1:
        st = hq.heappop(scoville)
        
        if st < K:
            cnt += 1                                 
            nd = hq.heappop(scoville)         
            hq.heappush(scoville, (st+(nd*2)))
        else:
            break
        
    return -1 if scoville[0] < K else cnt



# [1차] 프렌즈4블록(17679) - Lv.2
def solution(m, n, board):
    board.reverse()
    field = [list(cols) for cols in zip(*board)]
    answer = 0

    while True:
        bomb = set()
        
        for row in range(n - 1):
            for col in range(m - 1):
                try:
                    if field[row][col] == field[row + 1][col] == field[row][col + 1] == field[row + 1][col + 1]:
                        bomb.update({(row, col), (row + 1, col), (row, col + 1), (row + 1, col + 1)})
                except:
                    break

        if not len(bomb): break

        for r, c in bomb:
            field[r][c] = ''
            answer += 1

        for row in range(n): field[row] = list(''.join(field[row]))

    return answer