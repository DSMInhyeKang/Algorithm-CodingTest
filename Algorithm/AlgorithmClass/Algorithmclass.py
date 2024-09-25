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
visited = [False for _ in range(N+1)]

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



# 토마토(BFS 활용) - BOJ(7576)
from collections import deque

M, N = map(int, input().split())
box = [list(map(int, input().split())) for _ in range(N)]
dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]
period = 0
deq = deque()

for i in range(N) :
    for j in range(M) :
        if box[i][j]==1:
            deq.append((i,j,0))

while deq:
    ctr = deq.popleft()

    for k in range(4):
        x = ctr[0] + dx[i]
        y = ctr[1] + dy[i]

        if 0 <= x < n and 0 <= y < m and map[x][y]==0:
            map[x][y] = 1
            day = ctr[2] + 1
            deq.append((x,y,day))
print(day)



# 이분검색
N, M = map(int, input().split())
nums = list(map(int, input().split()))
nums.sort()

start, end = 0, N - 1
res = 0

while start <= end:
    mid = (start + end) // 2

    if nums[mid] == M:
        res = mid + 1
        break
    elif nums[mid] > M:
        end = mid - 1
    else:
        start = mid + 1

print(res)



# 랜선자르기(결정알고리즘) - BOJ 랜선 자르기(1654)
K, N = map(int, input().split())
cables = [int(input()) for _ in range(K)]

start, end = 1, max(cables)

while start <= end:
    mid = (start + end) // 2
    lines = 0

    for i in cables:
        lines += i // mid

    if lines >= N:
        start = mid + 1
    else:
        end =  mid - 1

print(end)



# 뮤직비디오(결정알고리즘)
N, M = map(int, input().split())
songs = list(map(int, input().split()))

start, end = 1, sum(songs)
storage = 0

while start <= end:
    mid = (start + end) // 2

    cnt, s = 1, 0

    for i in songs:
        if s + i > mid:
            cnt += 1
            s = i
        else:
            s += i
        
    if cnt <= M:
        storage = mid
        end = mid - 1
    else:
        start = mid + 1

print(storage)



# 회의실 배정(그리디) - BOJ 회의실 배정(1931)
n = int(input())
meetings = sorted([tuple(map(int, input().split())) for _ in range(n)], key=lambda x: (x[1], x[0]))
cnt = end = 0

for s, e in meetings:
    if s >= end:
        cnt += 1
        end = e

print(cnt)



# 씨름 선수(그리디)
N = int(input())
physical = sorted([tuple(map(int, input().split())) for _ in range(N)], key=lambda x: (x[1], x[0]))
print(physical)
cnt = N

for i in physical:
    for j in physical[physical.index(i)+1:]:
        if i[0] < j[0] and i[1] < j[1]:
            cnt -= 1
            break

print(cnt)



# 창고 정리
L = int(input())
heights = sorted(list(map(int, input().split())), key=lambda x: -x)
M = int(input())

for _ in range(M):
    heights[heights.index(max(heights))] -= 1
    heights[heights.index(min(heights))] += 1

print(max(heights) - min(heights))



# 역수열(그리디)
N = int(input())
inversed = list(map(int, input().split()))
origin = [0] * N

for i, v in enumerate(inversed):
    cnt = 0

    for j in range(N):
        if cnt == v and origin[j] == 0:
            origin[j] = i + 1
            break
        elif origin[j] == 0:
            cnt += 1

print(*origin)



# 마구간 정하기(결정 알고리즘)
import sys

N, C = map(int, sys.stdin.readline().split())
xi = sorted([int(input()) for _ in range(N)])
s, e = 1, xi[-1]
res = 0

while s <= e:
    mid = (s + e) // 2

    cnt = 1
    l = xi[0]

    for i in range(1, N):
    	if xi[i] - l >= mid:
            cnt += 1
            l = xi[i]
    
    if cnt >= C:
        res = mid
        s = mid + 1
    else:
        e = mid - 1

print(res)



# 침몰하는 타이타닉(그리디)
import sys
from collections import deque

N, M = map(int, sys.stdin.readline().split())
passengers = list(map(int, sys.stdin.readline().split()))
passengers = deque(sorted(passengers))
cnt = 0

while passengers:
    if len(passengers) == 1:
        cnt += 1
        break

    if passengers[0] + passengers[-1] > M:
        passengers.pop()
        cnt += 1
    else :
        passengers.popleft()
        passengers.pop()
        cnt += 1

print(cnt)



# 증가수열 만들기(그리디)
import sys

N = int(input())
seq = list(map(int, sys.stdin.readline().split()))
l, r = 0, N-1
tmp = []
last, order = 0, ''

while l <= r:
    if seq[l] > last:
        tmp.append((seq[l], 'L'))
    if seq[r] > last:
        tmp.append((seq[r], 'R'))

    tmp.sort()

    if len(tmp) == 0:
        break
    else:
        order += tmp[0][1]
        last = tmp[0][0]

        if tmp[0][1] == 'L':
            l += 1
        else:
            r -= 1

    tmp.clear()

print(len(order))
print(order)




# 큰 수 만들기
def solution(number, k):
    stack = []
    
    for n in number:
        # 스택이 비어있지 않고 k가 0보다 클 때(아직 다 제거하지 않았을 때) 마지막으로 들어간 값이 현재 숫자보다 작다면 
        while stack and k > 0 and stack[-1] < n:
            # 그 값을 pop
            stack.pop()
            # pop하면 수 하나를 제거한 것이므로 k 감소
            k -= 1
        # while문 조건에 해당 안 될 때 스택에 값 추가
        stack.append(n)
        
    # stack에 들어간 값 출력
    return ''.join(stack[:len(number)-k])


# 네트워크
def solution(n, computers):            
    def dfs(i):
        visited[i] = 1  # 방문 처리

        for a in range(n): 
            if computers[i][a] and not visited[a]: # i번 컴퓨터와 a번 컴퓨터가 연결되어 있고(1), a를 방문하지 않았을 때
                dfs(a)                             # a 노드 탐색


    answer = 0
    visited = [0 for i in range(len(n))]

    for i in range(n):
        if not visited[i]:  # 아직 i번째 노드를 방문하지 않았다면
            dfs(i)          # 탐색
            answer += 1
        
    return answer



# 카펫
import math
def solution(brown, yellow):
    w = ((brown+4)/2 + math.sqrt(((brown+4)/2)**2-4*(brown+yellow)))/2
    h = ((brown+4)/2 - math.sqrt(((brown+4)/2)**2-4*(brown+yellow)))/2
    return [w,h]

def solution(brown, yellow):
    answer = []
    total = brown + yellow                  # a * b = total
    for b in range(1,total+1):
        if (total / b) % 1 == 0:            # total / b = a
            a = total / b
            if a >= b:                      # a >= b
                if 2*a + 2*b == brown + 4:  # 2*a + 2*b = brown + 4 
                    return [a,b]
            
    return answer



# 입국 심사
def solution(n, times):
    answer = 0
    # right는 가장 비효율적으로 심사했을 때 걸리는 시간
    # 가장 긴 심사시간이 소요되는 심사관에게 n 명 모두 심사받는 경우이다.
    left, right = 1, max(times) * n
    while left <= right:
        mid = (left+ right) // 2
        people = 0
        for time in times:
            # people 은 모든 심사관들이 mid분 동안 심사한 사람의 수
            people += mid // time
            # 모든 심사관을 거치지 않아도 mid분 동안 n명 이상의 심사를 할 수 있다면 반복문을 나간다.
            if people >= n:
                break
        
        # 심사한 사람의 수가 심사 받아야할 사람의 수(n)보다 많거나 같은 경우
        if people >= n:
            answer = mid
            right = mid - 1
        # 심사한 사람의 수가 심사 받아야할 사람의 수(n)보다 적은 경우
        elif people < n:
            left = mid + 1
            
    return answer


# 소수 찾기(완전 탐색)
from itertools import permutations

def solution(n):
    a = set()  # 빈 집합 생성(중복 제거를 위해 집합 사용)
    
    for i in range(len(n)):  # n을 한 자리씩 나누었을 때
        a |= set(map(int, map("".join, permutations(list(n), i + 1))))  # 가능한 모든 순열을 원소로 갖는 집합과 a를 합집합(|) 한 결과를 a에 저장

    a -= set(range(0, 2))  # 집합 내부의 0, 1을 제거(소수가 아니므로)

    # 어떤 자연수의 약수끼리의 곱셉은 그 수의 제곱근을 기준으로 대칭으로 이루어짐. 따라서 max(a) + 1이 아닌 제곱근 max(a) + 1까지만 확인 
    for i in range(2, int(max(a) ** 0.5) + 1):  
        a -= set(range(i * 2, max(a) + 1, i))  # i의 배수들을 원소로 갖는 집합과 a를 차집합 연산(중복 제거)

    return len(a)



# 연속된 부분 수열의 합(178870) - Lv.2
def solution(sequence, k):
    l = r = 0
    bag = []
    prefixSum = [0] + [sum(sequence[:i + 1]) for i in range(len(sequence))] # prefix[r] - prefix[l] == (sequence[r] - sequence[l]) - 1

    while r < len(prefixSum):
        currentSum = prefixSum[r] - prefixSum[l]
        if currentSum == k:
            bag.append([l, r-1])  # prefix -> (0 ~ r-1) 합, 따라서 r-1
            l += 1
        elif currentSum < k:
            r += 1
        else:
            l += 1

    sortedArr = sorted(bag, key=lambda x: x[1] - x[0])

    return sortedArr[0]



# 무인도 여행(154540) - Lv.2
from collections import deque

def solution(maps):
    dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]
    visited = [[False] * len(maps[0]) for _ in range(len(maps))]
    answer, deq = [], deque()
    

    for i in range(len(maps)):
        for j in range(len(maps[0])):
            if maps[i][j] != 'X' and visited[i][j] == False:
                deq.append((i, j))
                visited[i][j] = True
                days = int(maps[i][j])

                while deq:
                    x, y = deq.popleft()
                    
                    for k in range(4):
                        nx = x + dx[k]
                        ny = y + dy[k]
                        
                        if 0 <= nx < len(maps) and 0 <= ny < len(maps[0]):
                            if maps[nx][ny] != 'X' and visited[nx][ny] == False:
                                deq.append((nx, ny))
                                visited[nx][ny] = True
                                days += int(maps[nx][ny])

                answer.append(days)

    return sorted(answer) if answer else [-1]



# 호텔 대실(155651) - Lv.2
from heapq import heappush, heappop

def solution(book_time):
    answer = 0
    heap = []
    booked = [(int(s[:2]) * 60 + int(s[3:]), int(e[:2]) * 60 + int(e[3:])) for s, e in book_time]
    
    for s, e in booked:
        if not heap:
            heappush(heap, e+10)
            continue

        if heap[0] <= s:
            heappop(heap)
        else:
            answer += 1

        heappush(heap, e+10)
    
    return answer



# 이중우선순위큐(42628) - Lv.3
from heapq import heapify, heappush, heappop

def solution(operations):
    heap = []
    
    for o in operations:
        alp, num = o.split()
        num = int(num)

        if alp == 'I':
            heappush(heap, num)    
        else:
            if heap:
                if num == -1:
                    heappop(heap)
                else:
                    heap.sort()
                    heap.pop()
                    
    heap.sort()
        
    return [heap[-1], heap[0]] if heap else [0, 0]



# 스타 수열(70130) - Lv.3
from collections import Counter

def solution(a):
    counter = Counter(a)
    answer = -1
    
    for e in counter:
        if counter[e] <= answer: continue
        
        cnt, idx = 0, 0
        
        while idx < len(a)-1:
            if (a[idx] == a[idx+1]) or (a[idx] != e and a[idx+1] != e): 
                idx += 1
                continue
            
            cnt += 1
            idx += 2
            
        answer = max(answer, cnt)
        
    return answer * 2