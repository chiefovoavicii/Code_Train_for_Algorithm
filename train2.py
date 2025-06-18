
#def fun_list()







str1 = "ddsa"
str2 = "laddsa"





def fun_commen(str1, str2):
    m = len(str1)
    n = len(str2)
    begin = 0
    maxlen = 0
    dp = [[False] * (n + 1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = True
                temp_i, temp_j = i, j
                while dp[i][j] == True:
                    i -= 1
                    j -= 1
                if temp_i - i + 1 > maxlen:
                    maxlen = temp_i - i + 1
                    begin = i
                i, j = temp_i, temp_j

    return str1[begin:begin+maxlen]
            
print(fun_commen(str1, str2))
print("\n")

class ListNode():
    def __init__(self, val = 0, next = None):
        self.next = next
        self.val = val
        
l1 = ListNode(1)
l2 = ListNode(2)
l3 = ListNode(3)
l1.next = l2
l2.next = l3
l3.next = l1    
def list_cycle(head):
    if not head or not head.next:
        return False
    
    slow = head
    fast = head.next

    while fast and fast.next and fast != slow:
        fast = fast.next.next
        slow = slow.next
    
    if fast == slow:
        return True
    if not fast or not fast.next:
        return False

print(list_cycle(l1))

    

    