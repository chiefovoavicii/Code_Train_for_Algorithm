from typing import List

candidate_str = input().strip()
candidate = list(map(int, candidate_str[1:-1].split(',')))
target = int(input().strip())


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

print(Solution().combinationSum2(candidate,target))