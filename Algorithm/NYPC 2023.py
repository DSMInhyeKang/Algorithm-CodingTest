# 메이플스토리 새로운 직업 고르기

n = int(input())
job = []

for _ in range(n):
    job.append(input())

k = int(input())
name = []

for _ in range(k):
    name.append(input())


idx = 0
result = n

for l in range(k):
    if name[l] in job:
        job.remove(name[l])


print(len(job))

for o in range(len(job)):
    print(job[o])