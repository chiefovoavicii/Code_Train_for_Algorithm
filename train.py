def fun(nums):
    n = len(nums)
    tails = [0] * n
    res = 0

    for num in nums:
        l, r = 0, res
        while l < r:
            m = (l+r)//2

            if num > tails[l]:
                l = m + 1
            else:
                r = m
        
        tails[l] = num
        if res == r:
            res += 1
    return res

nums = [1,2,4,2,3,5,6]
n = fun(nums)
print(n)