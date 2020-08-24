# DP

## 62. Unique Paths
**solution**
Approach 1: DP

- Time complexity: O(mn)

```java
class Solution {
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for(int i = 0; i < m; i++)
            dp[i][0] = 1;
        for(int j = 0; j < n; j++)
            dp[0][j] = 1;
        for(int i = 1; i < m; i++)
            for(int j = 1; j < n; j++)
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        return dp[m - 1][n - 1];
    }
    
}
```

Approach 2: Math    
result is (m + n - 2) C(n - 1)

```java
class Solution {
    public int uniquePaths(int m, int n) {
        return combine(m + n - 2, n - 1);
    }
    
    private int combine(int n, int m){
        if (m == 0) return 1;
        //avoid overflow
        m = Math.min(m , n - m);
        long a = 1, b = 1;
        for(int i = 0; i < m; i++){
            a *= (n - i);
            b *= (i + 1);
        }
        return (int) (a / b);
    }
    
   
}
```

## 70. Climbing Stairs
**solution**
Approach: DP

- Time complexity: O(n)
- Space complexity: O(n)

```java
public class Solution {
    public int climbStairs(int n) {
        if (n == 1) {
            return 1;
        }
        int first = 1;
        int second = 2;
        for (int i = 3; i <= n; i++) {
            int third = first + second;
            first = second;
            second = third;
        }
        return second;
    }
}
```
- Time complexity: O(n)
- Space complexity: O(1)

```java
class Solution {
    public int climbStairs(int n) {
    //f[0] = 1!!
    //pre1 is 1 step before, pre2 is 2 steps before
        int pre1 = 1, pre2 = 1;
        int res = 1;
        for(int i = 2; i < n + 1; i++){
            res = pre1 + pre2;
            pre2 = pre1;
            pre1 = res;
        }
        return res;
        
    }
    
}
```

## 64. Minimum Path Sum
**solution**
Approach: DP

```java
class Solution {
    public int minPathSum(int[][] grid) {
        int m = grid.length;
        if (m == 0) return 0;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for(int i = 1; i < m; i++)
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        for(int j = 1; j < n; j++)
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        for(int i = 1; i < m; i++)
            for(int j = 1; j < n; j++)
                dp[i][j] = Math.min(dp[i][j - 1], dp[i - 1][j]) + grid[i][j];
        return dp[m - 1][n - 1];
    }
    
}
```

## 72. Edit Distance
**solution** 
Approach: DP

- Time complexity: O(mn), where m, n is length of word1, word2
- Space complexity: O(mn)

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int l1 = word1.length(), l2 = word2.length();
        //add an empty char in the begin
        int[][] dp = new int[l1 + 1][l2 + 1];
        for(int i = 0; i < l1 + 1; i++)
            dp[i][0] = i;
        for(int j = 0; j < l2 + 1; j++)
            dp[0][j] = j;
        for(int i = 1; i < l1 + 1; i++){
            for(int j = 1; j < l2 + 1; j++){
                if (word1.charAt(i - 1) == word2.charAt(j - 1))
                //we can prove in this case, dp[i - 1][j - 1] <= dp[i][j - 1] + 1
                //and dp[i - 1][j - 1] <= dp[i - 1][j] + 1
                    dp[i][j] = dp[i - 1][j - 1];
                else
                    dp[i][j] = Math.min(dp[i][j - 1],
                        Math.min(dp[i - 1][j], dp[i - 1][j - 1])) + 1;
            }
        }
        return dp[l1][l2];
    }
}
```

- Time complexity: O(mn)
- Space complexity: O(n)

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int l1 = word1.length(), l2 = word2.length();
        //add an empty char in the begin
        int[] pre = new int[l2 + 1];
        int[] cur = new int[l2 + 1];
        for(int j = 0; j < l2 + 1; j++)
            pre[j] = j;
        for(int i = 1; i < l1 + 1; i++){
            cur[0] = i;
            for(int j = 1; j < l2 + 1; j++){
                if (word1.charAt(i - 1) == word2.charAt(j - 1))
               
                    cur[j] = pre[j - 1];
                else
                    cur[j] = Math.min(cur[j - 1],
                            Math.min(pre[j], pre[j - 1])) + 1;
            }
            int[] tmp = cur;
            cur = pre;
            pre = tmp;
                                      
        }
        return pre[l2];
    }
}
```

## 85. Maximal Rectangle
**solution**
Approach: Based on problem 84

```java
class Solution {
    public int maximalRectangle(char[][] matrix) {
        // regard as m 'Largest Rectangle in Histogram' problem
        int m = matrix.length;
        if (m == 0) return 0;
        int n = matrix[0].length;
        if (n == 0) return 0;
        int[] heights = new int[n];
        int res = 0;
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++)
                heights[j] = matrix[i][j] == '0' ? 0 : heights[j] + 1;
            res = Math.max(res, largestRectangleArea(heights));
        }
        return res;
    }
    private int largestRectangleArea(int[] heights) {
         if (heights == null || heights.length == 0) 
             return 0;
        
        //first index of the bar the left less than the current bar
        int[] lessFromLeft = new int[heights.length];
        //similarily
        int[] lessFromRight = new int[heights.length];
        lessFromLeft[0] = -1;
        lessFromRight[heights.length - 1] = heights.length;
        for(int i = 1; i < heights.length; i++){
            int p = i - 1;
            while(p >= 0 && heights[p] >= heights[i])
                p = lessFromLeft[p];
            lessFromLeft[i] = p;
        }
        for(int i = heights.length - 2; i >= 0; i--){
            int p = i + 1;
            while(p < heights.length && heights[p] >= heights[i])
                p = lessFromRight[p];
            lessFromRight[i] = p;
        }
        int maxArea = 0;
        for(int i = 0; i < heights.length; i++)
            maxArea = Math.max(maxArea, heights[i] * (lessFromRight[i] - lessFromLeft[i] - 1));
        return maxArea;
    }
}
```

## 139. Word Break

**solution**

Approach:  Top-down DP


```java
class Solution {
    
    public boolean wordBreak(String s, List<String> wordDict) {
        int[] memo = new int[s.length()];
        Arrays.fill(memo, -1);
        return helper(s, 0, wordDict, memo);
    }
    
    private boolean helper(String s, int begin,
                           List<String> wordDict,
                           int[] memo){
        if (begin == s.length()) return true;
        if (memo[begin] != -1) return memo[begin] == 1;
        
        boolean res = false;
        for(int i = 0; i < wordDict.size(); i++){
            if (s.startsWith(wordDict.get(i), begin)){
                res = res || 
                helper(s, begin + wordDict.get(i).length(),
                       wordDict, memo);
                if (res) break;
            }
            
        }
        memo[begin] = res ? 1 : 0;
        return res;
}
    
   
}
```

## 198. House Robber
**solution**
Approach: Top-down DP

- Time complexity: O(n)
- Space complexity: O(n)

```java
class Solution {
    public int rob(int[] nums) {
        int memo[] = new int[nums.length];
        Arrays.fill(memo, -1);
        return dp(0, nums, memo);
    }
    private int dp(int begin, int[] nums, int[] memo){
        if (begin >= nums.length) return 0;
        if (memo[begin] != -1) return memo[begin];
        
        int res = Math.max(nums[begin] + dp(begin + 2, nums, memo),
                          dp(begin + 1, nums, memo));
        memo[begin] = res;
        return res;
        
    }
}
```

## 221. Maximal Square
**solution**
Approach: DP

```java
class Solution {
    public int maximalSquare(char[][] matrix) {
        int m = matrix.length;
        if (m == 0) return 0;
        int n = matrix[0].length;
        int[][] dp = new int[m + 1][n + 1];
        int maxlen = 0;
        for(int i = 1; i < m + 1; i++){
            for(int j = 1; j < n + 1; j++){
                if (matrix[i - 1][j - 1] == '1'){
                    dp[i][j] = Math.min(dp[i - 1][j - 1],
                                     Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                maxlen = Math.max(maxlen, dp[i][j]);
                }
            }
        }
        return maxlen * maxlen;
    }
}
```

## 279. Perfect Squares
**solution**
Approach1 : DP

- Time Complexity: O(n * sqrt(n))(?)
- Space Complexity: O(n)
```java
class Solution {
    public int numSquares(int n) {
        //dp[n] = max{dp[n - i * i] + 1}(i * i <= n);
        int[] dp = new int[n + 1];
        dp[0] = 0;
        for(int i = 1; i < n + 1; i++){
            int min = Integer.MAX_VALUE;
            for(int j = 1; i - j * j >= 0; j++)
                min = Math.min(min, dp[i - j * j] + 1);
            dp[i] = min;
        }
        return dp[n];
        
    }
}
```
Approach2: Math


- Time Complexity: O(sqrt(n))
- Space Complexity: O(1)

四平方和定理: 任何数都可以由4个平方数组成，即 n = a^2 + b^2 + c^2 + d^2，所以这题的答案已经限定在了 [1,4] 之间

三平方和定理: n = a^2 + b^2 + c^2 当且仅当n 无法写成(4^k) * (8m + 7)(k,m为非负整数)

除了满足以上这个公式的数以外的任何数都可以由3个平方数组成

```java
class Solution {
    public int numSquares(int n) {
        int sqrt = (int) Math.sqrt(n);
        if (sqrt * sqrt == n) return 1;
        //if k == 2^i, then n % k = n & (k - 1)
        while((n & 3) == 0) n >>= 2;
        if ((n & 7) == 7) return 4;
        for(int i = 1; i <= sqrt; i++){
            int rest = n - i * i;
            int rs = (int) Math.sqrt(rest);
            if (rs * rs == rest) return 2;
        }
        return 3;
    }
}
```

## 322. Coin Change
**solution**

Approach 1 (Dynamic programming - Top down)


- Time Complexity: O(S * n), where S is the amount, n is denomination count
- Space Complexity: O(S)

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        //dp[i] = min(dp[i - coins[k]]) + 1 (dp[i - coins[k]] >= 0)
        //if i - coins[k] < 0 then dp[i - coins[j]] = -1
        int[] memo = new int[amount + 1];
        Arrays.fill(memo, -2);
        return helper(coins, amount, memo);
    }
    private int helper(int[] coins, int amount, int[] memo){
        if (amount < 0) return -1;
        if (amount == 0) return 0;
        if(memo[amount] != -2) return memo[amount];
        int res = Integer.MAX_VALUE;
        for(int i = 0; i < coins.length; i++){
            int submin = helper(coins, amount - coins[i], memo);
            if (submin >= 0)
                res = Math.min(res, submin + 1);
        }
        res = (res == Integer.MAX_VALUE) ? -1 : res;
        memo[amount] = res;
        return res;
    }
}
```

Approach 2 (Dynamic programming - Bottom up)

- Time Complexity: O(S * n), where S is the amount, n is denomination count
- Space Complexity: O(S)

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        for(int i = 1; i < amount + 1; i++){
            int min = Integer.MAX_VALUE;
            for(int j = 0; j < coins.length; j++){
                int pre = i - coins[j] < 0
                    ? -1 : dp[i - coins[j]];
                if (pre >= 0)
                    min = Math.min(min, 1 + pre);
                }
            min = (min == Integer.MAX_VALUE) ? -1 : min;
            dp[i] = min;
        }
        return dp[amount];
    }
}
```
## 416. Partition Equal Subset Sum
**solution**

- Time Complexity: O(n * sum)
- Space Complexity: O(sum)

```java
class Solution {
    public boolean canPartition(int[] nums) {
        //f(n, s) -> in [0, n - 1], if there exists
        // a subset whose sum is s;
        //f(n, s) = f(n - 1, s) || f(n - 1, s - nums[n - 1]);
        int sum = 0;
        for(int num : nums)
            sum += num;
        if (sum % 2 != 0) return false;
        int s = sum / 2;
        boolean[] dp = new boolean[s + 1];
        dp[0] = true;
        for(int num : nums){
            //decrese j, if dp[j] is changed
            //then the value of dp[k](k < j) doesn't don't 
            //require the original value in dp[j].
            //but if we increse j, the value of dp[k](k > j)
            //require the original value in dp[j], but dp[j]
            //is changed, we will get wrong answer.
            //if we want to increase j, use an extra array to save the current result
            for(int j = s; j >= 0; j--){
                if (j - num >= 0)
                    dp[j] = dp[j] || dp[j - num];
            }
            
        }
        return dp[s];
        
    }
}
```

## 494. Target Sum
**solution**
Approach: DP
dp[i][j] = dp[i-1][j+nums[i]] + dp[i-1][j-nums[i]]
dp[i][j] ==> ways of subarray [0, i] that assign symbols to make sum in the subarray equal to j;

- Time Complexity: O(n * sum)
- Space Complexity: O(sum)


```java
class Solution {
    public int findTargetSumWays(int[] nums, int S) {
        int sum = 0;
        for(int num : nums)
            sum += num;
        if (S < -sum || S > sum) return 0;
        int[] dp = new int[2 * sum + 1];
        //0 ~ 2 * sum <====> -sum ~ sum
        //empty subarray in the initial state
        dp[sum] = 1;
        for(int i = 0; i < nums.length; i++){
            int[] tmp = new int[2 * sum + 1];
            for(int j = 0; j < 2 * sum + 1; j++){
                //dp[j] > 0 to avoid index out of range
                if (dp[j] > 0){
                    tmp[j - nums[i]] += dp[j];
                    tmp[j + nums[i]] += dp[j];
                }
            }
            dp = tmp;
        }
        return dp[S + sum];
    }
}
```