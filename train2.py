
#def fun_list()







str1 = "ddsawsda"
str2 = "laddsaa"




def fun_commen(str1, str2):
    m = len(str1)
    n = len(str2)
    end_idx = 0
    
    maxlen = 0
    dp = [[0] * (n + 1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
                if dp[i][j] > maxlen:
                    maxlen = dp[i][j]
                    end_idx = i
                

    return str1[end_idx - maxlen:end_idx]
            
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

    

    