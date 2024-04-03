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