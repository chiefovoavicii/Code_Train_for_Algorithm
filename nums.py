from typing import List

candidate = list(map(int,input().split()))
target = int(input().strip())

text = input().strip()  # 读取一行文本并去除首尾空格
print(type(text))  

num = int(input().strip())  # 转换为整数
print(type(num))            # 输出：<class 'int'>

data_str = input().strip()  # 如："10 20 30"
num_list = list(map(int, data_str.split())) 

csv_data = input().strip()  # 如："apple,banana,orange"
str_list = csv_data.split(',')

class Solution:
    def combinationSum2(self, candidates, target):
        candidates.sort()
        n = len(candidates)
        res =[]
        path = []

        def dfs(i,target):
            if target == 0:
                res.append(path[:])
                return
            
            for j in range(i,n):
                if target < candidates[j]:
                    break
                if j > i and candidates[j] == candidates[j-1]:
                    continue
                
                path.append(candidates[j])
                dfs(j+1, target - candidates[j])
                path.pop()

        dfs(0, target)
        return res

print(candidate)