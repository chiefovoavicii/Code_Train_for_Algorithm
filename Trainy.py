#500 囚犯 一排 一次报数 每次随机奇数会随机铅笔  下一次 奇数随机墙壁 最后一个人无罪释放 加入我是最后囚犯 大概率货到最后

import random


def fun(i):
    nums = [0] * 500
    for j in range(1,501):
        nums[j-1] = j
    nums[i] = 0
    cnt = 0
    n = 500
    while n:
        x = random.randint(0, n)
        if x % 2:
            cnt += 1
            res = nums[x]
            nums.pop(x)
            n -= 1
            if not res:
                return cnt 
            else:
                continue
        

def solve():
    ans_list = []
    for i in range(500):
        time = 1000
        temp_list = []
        while time: 
            temp = fun(i)
            temp_list.append(temp)
            time -= 1
        ans = sum(temp_list)/1000
        ans_list.append(ans)
    maxans = 0
    for i,val in enumerate(ans_list):
        if val > maxans:
            final = (i,val)
    return final[0]

solve()


            

