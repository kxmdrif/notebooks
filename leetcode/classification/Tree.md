# Tree

## 94. Binary Tree Inorder Traversal

Approach 1: Recursive Approach

- Time complexity: O(n)
- Space complexity: worst : O(n), average: O(logn)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        traversal(res, root);
        return res;
    }
    
    private void traversal(List<Integer> res, TreeNode root){
        if (root == null) return;
        traversal(res, root.left);
        res.add(root.val);
        traversal(res, root.right);
    }
}
```

Approach 2: Iterating method using Stack

- Time complexity: O(n)
- Space complexity: worst : O(n), average: O(logn)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while(cur != null || !stack.isEmpty()){
            while(cur != null){
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            res.add(cur.val);
            cur = cur.right;
        }
        return res;
    }
    
    
}
```

## 96. Unique Binary Search Trees

```java
class Solution {
    
    //f(n) = f(0)f(n - 1) + f(1)f(n - 2) + ... + f(n - 2)f(1) + f(n - 1)f(0)
    //f(0) = f(1)= 1, f(k)f(n - k - 1) --> k + 1 is root
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 1;
        for(int i = 2; i < n + 1; i++)
            for(int j = 0; j < i; j++)
                dp[i] += dp[j] * dp[i - j - 1];
        return dp[n];
    }
   
}
```

## 98. Validate Binary Search Tree

**solution**
Approach 1: Recursion

- Time Complexity : O(n)
- Space Complexity: O(n)

```java
class Solution {
    private boolean res;
    public boolean isValidBST(TreeNode root) {
        return helper(root, null, null);
    }
    //use integer instead of int to avoid root.val = INT_MAX OR INT_MIN
    private boolean helper(TreeNode root, Integer lower, Integer upper){
        if (root == null) return true;
        if (lower != null && root.val <= lower) return false;
        if (upper != null && root.val >= upper) return false;
        
        return helper(root.left, lower, root.val)
            && helper(root.right, root.val, upper);
    }
    
    
}
```

Approach 2: Inorder traversal

- Time Complexity : O(n)
- Space Complexity: O(n)

```java
class Solution {
    private boolean res;
    public boolean isValidBST(TreeNode root) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        //use long instead of int to avoid root.val = INT_MAX OR INT_MIN
        long pre = Long.MIN_VALUE;
        while(cur != null || !stack.isEmpty()){
            while(cur != null){
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            if (cur.val <= pre) return false;
            pre = cur.val;
            cur = cur.right;
            
        }
        return true;
        
    }
    
    
}
```

## 101. Symmetric Tree

**solution**

```java
class Solution {
    //Symmetric Tree  ==>  the tree is the mirror ofitself
    public boolean isSymmetric(TreeNode root) {
        return isMirror(root, root);
    }
    
    private boolean isMirror(TreeNode t1, TreeNode t2){
        if (t1 == null && t2 == null) return true;
        if (t1 == null || t2 == null) return false;
        
        return t1.val == t2.val && isMirror(t1.left, t2.right)
            && isMirror(t1.right, t2.left);
    }
}
```

## 102. Binary Tree Level Order Traversal

**solution**

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
import java.util.*;
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) return res; 
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        
        while(!queue.isEmpty()){
            int size = queue.size();
            List<Integer> level = new ArrayList<>();
            for(int i = 0; i < size; i++){
                TreeNode node = queue.poll();
                level.add(node.val);
                if (node.left != null) queue.offer(node.left);
                if (node.right != null) queue.offer(node.right);
            }
            res.add(level);
        }
        return res;
    }
}
```

## 104. Maximum Depth of Binary Tree

**solution**
Approach1 : Recursion

```java
import java.util.*;

class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }
}
```
Apoproach2: Iteration

```java
import java.util.*;

class Solution {
    public int maxDepth(TreeNode root) {
        if (root == null) return 0;
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.offer(root);
        int depth = 0;
        while(!queue.isEmpty()){
            int size = queue.size();
            for(int i = 0; i < size; i++){
                TreeNode node = queue.poll();
                if (node.left != null)
                    queue.offer(node.left);
                if (node.right != null)
                    queue.offer(node.right);
            }
            depth++;
        }
        return depth;
        
    }
}
```

## 105. Construct Binary Tree from Preorder and Inorder Traversal
**solution**

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        Map<Integer, Integer> inMap = new HashMap<>();
        for(int i = 0; i < inorder.length; i++)
            inMap.put(inorder[i], i);
        return helper(inMap, preorder, inorder, 0, 0, preorder.length);
    }
    
    private TreeNode helper(Map<Integer, Integer> inMap, 
                            int[] preorder, int[] inorder,
                int preStart, int inStart, int length){
        if (length == 0) return null;
        int rootval = preorder[preStart];
        TreeNode root = new TreeNode(rootval);
        //use map instaed of scan to find the position of root
        int idx = inMap.get(rootval);
        
        root.left = helper(inMap, preorder, inorder,
                           preStart + 1, inStart, idx - inStart);
        root.right = helper(inMap, preorder, inorder,
                           preStart + idx - inStart + 1,
                        idx + 1, length - idx + inStart - 1);
        return root;
    }
    
}
```

## 114. Flatten Binary Tree to Linked List
**solution**
Approach 1

- Time Complexity: O(n^2)

```java
class Solution {
    public void flatten(TreeNode root) {
        if(root == null) return;
        TreeNode left = root.left;
        TreeNode right = root.right;
        flatten(left);
        flatten(right);
        root.right = left;
        root.left = null;
        TreeNode ptr = root;
        while(ptr.right != null)
            ptr = ptr.right;
        ptr.right = right;
        
    }
    
}
```

Approach 2

- Time Complexity: O(n)

```
    1
   / \
  2   5
 / \   \
3   4   6
-----------        
pre = 5
cur = 4

    1
   / \
  2   \
 / \  |
3   4 |
     \|
      5
       \
        6
-----------        
pre = 4
cur = 3

    1
   / \
  2  |
 /|  | 
3 |  |
 \|  |
  4  |
   \ |
    5
     \
      6
-----------        
cur = 2
pre = 3

    1
   / \
  2   \
   \   \
    3   \
     \  |
      4 |
       \|
        5
         \
          6
-----------        
cur = 1
pre = 2

1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
```

```java
class Solution {
    private TreeNode prev = null;
    public void flatten(TreeNode root) {
        if (root == null) return;
        flatten(root.right);
        flatten(root.left);
        root.right = prev;
        root.left = null;
        prev = root;
    }
    
}
```

## 124. Binary Tree Maximum Path Sum

**solution**

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    private int max;
    public int maxPathSum(TreeNode root) {
        max = Integer.MIN_VALUE;
        travel(root);
        return max;
    }
    private int travel(TreeNode root){
        if (root == null) return 0;
        int leftVal = travel(root.left);
        int rightVal = travel(root.right);
        int maxRoot = Math.max(root.val, root.val + leftVal);
        maxRoot = Math.max(maxRoot, root.val + rightVal);
        maxRoot = Math.max(maxRoot, root.val + rightVal + leftVal);
        
        if (maxRoot > max)
            max = maxRoot;
        return Math.max(root.val, Math.max(root.val + leftVal, root.val + rightVal));
    }
}

-----------------or------------------------------
class Solution {
    private int max;
    public int maxPathSum(TreeNode root) {
        max = Integer.MIN_VALUE;
        travel(root);
        return max;
    }
    private int travel(TreeNode root){
        if (root == null) return 0;
        int left = Math.max(0, travel(root.left));
        int right = Math.max(0, travel(root.right));
        max = Math.max(max, root.val + left + right);
        return root.val + Math.max(left, right);
        
        
    }
}
```

## 226. Invert Binary Tree
**solution**
Approach1 : Recursive

- Time complexity: O(n)
- Space complexity: O(h), h is the height of the tree

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return root;
        invertTree(root.left);
        invertTree(root.right);
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
        return root;
    }
}
```
Approach2 : Iterative

- Time complexity: O(n)
- Space complexity: O(n)

```java
class Solution {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            TreeNode tmp = node.left;
            node.left = node.right;
            node.right = tmp;
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }
        return root;
    }
}
```

## 230. Kth Smallest Element in a BST
**solution**
Approach 1: Recursion
```java
class Solution {
    private int res;
    private int count;//用Integer也可
    public int kthSmallest(TreeNode root, int k) {
        this.res = 0;
        this.count = k;
        inOrder(root);
        return res;
    }
    private void inOrder(TreeNode root){
        if (root == null) return;
        inOrder(root.left);
        count = count - 1;
        if (count == 0){
            res = root.val;
            return;
        }
        inOrder(root.right);
    }
}

/*
class Solution {
    private int res;
    public int kthSmallest(TreeNode root, int k) {
        res = 0;
        inOrder(root, k);
        return res;
    }
    不能这样写因为Integer对象k自增/自减或者k = k - 1这种
    执行后引用k会指向新的位置,原来的对象没有修改.违背了每次递归
    都修改同一个对象的想法。
    private void inOrder(TreeNode root, Integer k){
        if (root == null) return;
        inOrder(root.left, k);
        k = k - 1;
        if (k == 0) res = root.val;
        inOrder(root.right, k);
    }
}
*/
```
Approach 2: Iteration
```java
class Solution {
    public int kthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<>();
        TreeNode cur = root;
        while(cur != null || !stack.isEmpty()){
            while(cur != null){
                stack.push(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            k--;
            if (k == 0) return cur.val;
            cur = cur.right;
        }
        return -1;
    }
    
}
```

## 236. Lowest Common Ancestor of a Binary Tree
**solution**

- Time complexity: O(n)
- Space complexity: O(n)
```java
class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q)
            return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        return left == null ? right : right == null ? left : root;
    }
}
```

## 297. Serialize and Deserialize Binary Tree
**solution**

Approach 1: DFS

- Time Complexity: O(n)
- Space Complexity: O(n)

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class Codec {
    private static final String SPLITTER = ",";
    private static final String NN = "#";

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        buildString(root, sb);
        return sb.toString();
    }
    
    private void buildString(TreeNode root, StringBuilder sb){
        if (root == null){
            sb.append(NN).append(SPLITTER);
            return;
        }
        sb.append(root.val).append(SPLITTER);
        buildString(root.left, sb);
        buildString(root.right, sb);
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        //note: for example
        //"a,".split(",") ==> len = 1
        //",a".split(",") ==> len = 2
        Queue<String> queue = new LinkedList<>();
        queue.addAll(Arrays.asList(data.split(",")));
        return buildTree(queue);
    }
    
    private TreeNode buildTree(Queue<String> queue){
        String node = queue.poll();
        if (node.equals(NN)){
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(node));
        root.left = buildTree(queue);
        root.right = buildTree(queue);
        return root;
    }
}

// Your Codec object will be instantiated and called as such:
// Codec codec = new Codec();
// codec.deserialize(codec.serialize(root));
```

Approach 2: BFS

- Time Complexity: O(n)
- Space Complexity: O(n)

```java
public class Codec {
    private static final String SPLITTER = ",";
    private static final String NN = "#";
    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        if (root == null) return "";
        StringBuilder sb = new StringBuilder();
        //note: linkedlist can add null to the queue
        //but ArrayDeque can't
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            if (node == null){
                sb.append(NN).append(SPLITTER);
                continue;
            }
            sb.append(node.val).append(SPLITTER);
            queue.offer(node.left);
            queue.offer(node.right);
        }
        return sb.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        if (data.isEmpty()) return null;
        Queue<TreeNode> queue = new LinkedList<>();
        String[] items = data.split(SPLITTER);
        int ptr = 0;
        TreeNode root = new TreeNode(Integer.parseInt(items[ptr++]));
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            String item = items[ptr++];
            if (!item.equals(NN)){
                node.left = new TreeNode(Integer.parseInt(item));
                queue.offer(node.left);
            }
            item = items[ptr++];
            if (!item.equals(NN)){
                node.right = new TreeNode(Integer.parseInt(item));
                queue.offer(node.right);
            }
        }
        return root;
        /* This way is also ok
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            if (node == null) continue;
            String item = items[ptr++];
            if (!item.equals(NN)){
                node.left = new TreeNode(Integer.parseInt(item));
                
            }
            queue.offer(node.left);
            item = items[ptr++];
            if (!item.equals(NN)){
                node.right = new TreeNode(Integer.parseInt(item));
                
            }
            queue.offer(node.right);
        }
        */
        
    }
}
```

## 337. House Robber III
**solution**

- Time Complexity: O(n)
- Space Complexity: O(logn)


```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public int rob(TreeNode root) {
        int[] res = robSub(root);
        return Math.max(res[0], res[1]);
    }
    //note: we return both res[0] and res[1] instead of the max of them only
    //because the parent node need them both to calculate 
    private int[] robSub(TreeNode root){
        //res[0] -> max value without robbing current node
        //res[1] -> max value with robbing current node
        int[] res =  new int[]{0, 0};
        if (root == null) return res;
        int[] leftMax = robSub(root.left);
        int[] rightMax = robSub(root.right);
        //if we don't rob current node, then we can choose to rob
        //the sub-node or not.
        res[0] = Math.max(leftMax[0], leftMax[1]) + 
            Math.max(rightMax[0], rightMax[1]);
        //if we rob the currentNode, then we can onlty choose to
        //not rob the sub-node
        res[1] = root.val + leftMax[0] + rightMax[0];
        return res;
    }
}
```

## 437. Path Sum III
**solution**

Approach 1: Use HashMap

- Time Complexity: O(n)
- Space Complexity: O(logn)


Each recursion returns the total count of valid paths in the subtree rooted at the current node. And this sum can be divided into three parts:
- the total number of valid paths in the subtree rooted at the current node's left child
- the total number of valid paths in the subtree rooted at the current node's right child
- the number of valid paths ended by the current node

Similar to `560. Subarray sum equals K problem`

```java
class Solution {
    public int pathSum(TreeNode root, int sum) {
        Map<Integer, Integer> preSum = new HashMap<>();
        //Default sum = 0 has one count
        preSum.put(0, 1);
        return helper(root, 0, sum, preSum);
    }
    public int helper(TreeNode cur, int sum, int target,
                     Map<Integer, Integer> preSum){
        if (cur == null) return 0;
        // update the prefix sum by adding the current val
        sum += cur.val;
        // get the number of valid path, ended by the current node
        //example: 1->2->3->4, cur = 4, target = 7, sum = 10
        //sum - target = 3 exists in {0:1,1:1,3:1,6:1}
        //so 3->4 is a valid path
        int res = preSum.getOrDefault(sum - target, 0);
        // update the map with the current sum, 
        //so the map is good to be passed to the next recursion
        preSum.put(sum, preSum.getOrDefault(sum, 0) + 1);

        res += helper(cur.left, sum, target, preSum)
            + helper(cur.right, sum, target, preSum);
        // restore the map, as the recursion goes from the bottom to the top
        //Remove the current node so it wont affect other path
        preSum.put(sum, preSum.getOrDefault(sum, 0)  - 1);
        return res;
        
    }
}
```

Approach 2: DFS

- Time Complexity: O(n^2) in worst case (no branching); O(nlogn) in best case (balanced tree).
- Space Complexity: O(n) due to recursion

```java
class Solution {
    public int pathSum(TreeNode root, int sum) {
        // pathSum(root, sum) returns number of paths in the subtree rooted at root
        // s.t. the sum of values on the path equals `sum`. 
        // The path might not include root.
        if (root == null) return 0;
        return dfs(root, sum) + pathSum(root.left, sum)
            + pathSum(root.right, sum);
    }
    public int dfs(TreeNode root, int sum){
        // pathSumFrom(root, sum) returns number of paths INCLUDING
        //(not necessary starting from)
        // root whose sum of values equals `sum`.
        if (root == null) return 0;
        return (sum == root.val ? 1 : 0)
            + dfs(root.left, sum - root.val)
            + dfs(root.right, sum - root.val);
    }
}
```

## 543. Diameter of Binary Tree

**solution**

Approach: Similar to 124. Binary Tree Maximum Path Sum, here we regard the value of each node as 1

- Time Complexity: O(n)
- Space Complexity: O(n)(worst case. totally not balanced)


```java
class Solution {
    private int max;
    public int diameterOfBinaryTree(TreeNode root) {
        //if we don't add this, root = null will return -1
        if (root == null) return 0;
        max = Integer.MIN_VALUE;
        travel(root);
        //length of path equals to pathsum - 1(regard the value of each node as 1)
        return max - 1;
    }
    
    public int travel(TreeNode root){
        if (root == null) return 0;
        int left = travel(root.left);
        int right = travel(root.right);
        max = Math.max(max, left + right + 1);
        return Math.max(left + 1, right + 1);
    }
}
```
