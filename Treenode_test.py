

class TreeNode:
    def __init__(self, val=0, left = None, right = None):
        self.val = val
        self.left = None
        self.right = None


class Solution:
    def isValidBST(self, root):
        self.last = -float("inf")
        def dfs(root):
            if not root:
                return True
            if not dfs(root.left):     #只有这样实现遍历才能够把False传导最外层
                return False
            
            if root.val <= self.last:
                return False
            else:
                self.last = root.val

            if not dfs(root.right):
                return False

            return True
        

        return dfs(root)
######构造二叉树
def list_to_Tree(nums):
    if not nums or nums[0] is None:
        return None
    i = 1
    root = TreeNode(nums[0])
    queue = [root]

    while queue and i < len(nums):
        node = queue.pop(0)

        if nums[i] is not None and i < len(nums):
            node.left = TreeNode(nums[i])
            queue.append(node.left)
        i += 1

        if nums[i] is not None and i < len(nums):
            node.right = TreeNode(nums[i])
            queue.append(node.right)
        i += 1

    return root

input_str = input().strip()  #获取输入并去除前后空白符
print(input_str)
elements = []
for c in input_str.split(','):
    if c.lower() == 'null':
        elements.append(None)
    else:
        elements.append(int(c))





# root = list_to_Tree(elements)

# print(Solution().isValidBST(root))



class Treenode():
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None

def nums_to_tree(nums):
    

    def dfs(root,i):
        if len(nums) < 2*i + 2:
            return False
        root = Treenode(root)
        root.left = dfs(nums[2*i+1],2*i+1)
        root.right = dfs(nums[2*i+2],2*i+2)
        return root

    return dfs(nums[0],0)

nums = [1,2,3,4,5,6,7]
tree = nums_to_tree(nums)
print(tree.left.val)

    
