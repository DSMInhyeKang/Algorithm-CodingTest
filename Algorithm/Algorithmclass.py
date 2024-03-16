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



# 요세푸스 문제
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