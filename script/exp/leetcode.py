import collections
from optparse import Option
from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(8)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

# 容器最大装水量
def maxContains(height: List[int]) -> int:
    left, right = 0, len(height) - 1
    ans = 0
    while left < right:
        h = right - left
        if height[left] < height[right]:
            target = height[left] * h
            left += 1
        else:
            target = height[right] * h
            right -= 1
        ans = target if target > ans else ans
    return ans

# 三数相加
def threeSum(nums: List[int]) -> List[List[int]]:
    ans = list()
    n = len(nums)

    nums.sort()
    for first_idx in range(n):
        second_idx = first_idx + 1
        third_idx = n - 1
        if first_idx > 0 and nums[first_idx] == nums[first_idx - 1]:
            continue
        if nums[first_idx] + nums[first_idx + 1] + nums[first_idx + 2] > 0:
            break
        if nums[first_idx] + nums[-2] + nums[-1] < 0:
            continue
        while second_idx < third_idx:
            s = nums[first_idx] + nums[second_idx] + nums[third_idx]
            if s > 0:
                third_idx -= 1
            elif s < 0:
                second_idx += 1
            else:
                ans.append([nums[first_idx], nums[second_idx], nums[third_idx]])
                second_idx += 1
                while second_idx < third_idx and nums[second_idx] == nums[second_idx - 1]:
                    second_idx += 1
                third_idx -= 1
                while second_idx < third_idx and nums[third_idx] == nums[third_idx + 1]:
                    third_idx -= 1
    return ans

# 深度优先
def dfs(root):
    if root is None:
        return
    dfs(root.left)
    dfs(root.right)
    print(root.val)


# 广度优先
def bfs(root):
    ans = []
    q = collections.deque()
    if root is None:
        return
    q.append(root)
    while q:
        for _ in range(len(q)):
            node = q.popleft()
            ans.append(node.val)
            if node.left is not None: q.append(node.left)
            if node.right is not None: q.append(node.right)
    print(ans)

# 给先序重构二叉搜索树
def createTreeByFirst(nodes):
    if not nodes:
        return
    root_val = nodes[0]
    root = TreeNode(root_val)
    left_nodes = [x for x in nodes if x < root_val]
    right_nodes = [x for x in nodes if x > root_val]
    root.left = createTreeByFirst(left_nodes)
    root.right = createTreeByFirst(right_nodes)
    return root

# 给先序和中序构造二叉树
def formatTreeByPreMid(preNodes: List, inNodes: List):
    if not preNodes:
        return
    root_value = preNodes[0]
    root = TreeNode(root_value)
    idx = inNodes.index(root_value)
    left_pre_nodes = preNodes[1:idx]
    right_pre_nodes = preNodes[idx+1:]
    left_in_nodes = inNodes[:idx]
    right_in_nodes = inNodes[idx+1:]
    root.left = formatTreeByPreMid(left_pre_nodes, left_in_nodes)
    root.right = formatTreeByPreMid(right_pre_nodes, right_in_nodes)
    return root

# 逆时针
def spiralOrderA(matrix: List[List[int]]) -> List[int]:
    res = list()
    if not matrix or not matrix[0]:
        return res

    up, down, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
    while left <= right and up <= down:
        for x in range(right, left-1, -1):
            res.append(matrix[up][x])
        up += 1
        if up < down:
            for y in range(up, down + 1):
                res.append(matrix[y][left])
            left += 1
            if left < right:
                for y in range(left, right + 1):
                    res.append(matrix[down][y])
                down -= 1
                if up < down:
                    for x in range(down, up-1, -1):
                        res.append(matrix[x][right])
                    right -= 1
    return res[::-1]

# 顺时针
def spiralOrderT(matrix: List[List[int]]) -> List[int]:
    res = list()
    if not matrix or not matrix[0]:
        return res

    up, down, left, right = 0, len(matrix)-1, 0, len(matrix[0])-1
    while left <= right and up <= down:
        for x in range(left, right+1):
            res.append(matrix[up][x])
        up += 1
        if up <= down:
            for y in range(up, down+1):
                res.append(matrix[y][right])
            right -= 1
            if left <= right:
                for y in range(right, left - 1, -1):
                    res.append(matrix[down][y])
                down -= 1
                if up <= down:
                    for x in range(down, up - 1, -1):
                        res.append(matrix[x][left])
                    left += 1
    return res

# 反转链表
def reverseChain(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    if not head:
        return head

    cnt = 0
    cur = head
    while cur:
        cnt += 1
        cur = cur.next
    rt = k % cnt
    if rt == 0:
        return head
    fast, slow = head, head
    for _ in range(rt):
        fast = fast.next
    while fast.next:
        fast = fast.next
        slow = slow.next
    fast.next = head
    head = slow.next
    slow.next = None
    return head

def findMin(nums:List[int]) -> int:
    if not nums:
        return 0
    left, right = 0, len(nums)-1
    while left < right:
        pivot = left + (right - left) // 2
        if nums[pivot] > nums[right]:
            left = pivot + 1
        else:
            right = pivot
    return nums[left]

def mergeIntervals(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort(key= lambda x: x[0])
    ans = list()

    for interval in intervals:
        if ans and ans[-1][1] > interval[0]:
            ans[-1] = [ans[-1][0], max(ans[-1][1], interval[1])]
        else:
            ans.append(interval)

    return ans

def generateBuckets(n):

    ans = []
    def reverseLoop(s, left, right):

        if len(s) == 2 * n:
            ans.append(s)
            return
        if left < n:
            reverseLoop(s + '(', left+1, right)
        if right < left:
            reverseLoop(s + ')', left, right+1)
    reverseLoop('', 0, 0)
    return ans

# 岛屿数量
def numIslands(grid):

    def dfs(grid, row, column):
        grid[row][column] = 0
        rows, columns = len(grid), len(grid[0])
        l = [(row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)]
        for x, y in l:
            if 0 <= x < rows and 0 <= y < columns and grid[x][y] == '1':
                dfs(grid, x, y)

    if len(grid) == 0:
        return 0
    rows, columns = len(grid), len(grid[0])
    ans = 0
    for i in range(rows):
        for j in range(columns):
            if grid[i][j] == '1':
                ans += 1
                dfs(grid, i, j)
    return ans


if __name__ == '__main__':
    # dfs(root)
    # bfs(root)
    # p = createTreeByFirst([5, 2, 3, 4, 6])
    # print(p)
    # print(spiralOrder([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    # print(spiralOrderT([[1, 2], [3, 4]]))
    # head = ListNode(1)
    # head.next = ListNode(2)
    # head.next.next = ListNode(3)
    # head.next.next.next = ListNode(4)
    # head.next.next.next.next = ListNode(5)

    # head = ListNode(0)
    # head.next = ListNode(1)
    # head.next.next = ListNode(2)
    # nums = [4, 5, 6, 7, 0, 1, 2]
    # findMin(nums)
    # print(reverseChain(head, 2))

    # print(maxContains([1, 8, 6, 2, 5, 4, 8, 3, 7]))
    # nums = [-2, 0, 1, 1, 2]
    # print(threeSum(nums))
    # n = 3
    # print(generateBuckets(n))
    grid = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]
    print(numIslands(grid))
