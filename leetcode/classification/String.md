# String

## 3. Longest Substring Without Repeating Characters

**solution**

Time complexity : O(n)
Space complexity (Table): O(m). m is the size of the charset.


int[26] for Letters 'a' - 'z' or 'A' - 'Z'
int[128] for ASCII
int[256] for Extended ASCII

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        int[] index = new int[128];
        int res = 0;
        for(int i = 0, j = 0; j < n; j++){
            //if s[j] has a duplicated in [i, j) with index j1,
            //we let i = j1 + 1, instead of increasing i little by little
            //if s[j] dosen't have a duplicated in [i, j) 
            //(index[s.charAt(j)] = 0 or <= i), then i = i.
            i = Math.max(index[s.charAt(j)], i);
            res = Math.max(res, j - i + 1);
            index[s.charAt(j)] = j + 1;
        }
        return res;
    }
}
```

## 5. Longest Palindromic Substring

**solution**
- Time complexity: O(n^2)
- Space complexity: O(1)

```java
class Solution {
    public String longestPalindrome(String s) {
        if (s == null || s.isEmpty()) return "";
        int start = 0, end = 0;
        
        for(int i = 0; i < s.length(); i++){
            int len1 = appendCenter(s, i, i);
            int len2 = appendCenter(s, i, i + 1);
            int len = Math.max(len1, len2);
            if (len > end - start){
                start = i - (len - 1) / 2;
                end = i + len / 2;
            }
        }
        return s.substring(start, end + 1);
    }
    
    private int appendCenter(String s, int l, int r){
        while(l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)){
            l--;
            r++;
        }
        return r - l - 1;
    }
}
```

## 6. ZigZag Conversion
**solution**
- Time Complexity: O(n) where n = len(s)
- Space Complexity: O(n)

```java
class Solution {
    /*
    T = 2 * numRows - 2
    row 0 : indexes: k * T
    row numRows-1: indexes: numRows - 1 + k * T
    row i: indexes: i + k * T and T - i + k * T
    */
    public String convert(String s, int numRows) {
        if (numRows == 1) return s;
        int n = s.length();
        StringBuilder res = new StringBuilder();
        int cycle = 2 * numRows - 2;
        for(int i = 0; i < numRows; i++){
            for(int k = 0; i + k * cycle < n; k++){
                res.append(s.charAt(i + k * cycle));
                if (i != 0 && i != numRows - 1 &&
                    cycle - i + k * cycle < n)
                    res.append(s.charAt(cycle - i + k * cycle));
            }
        }
        return res.toString();
    }
}
```

## 10. Regular Expression Matching

**solution**
Approach DP

- Time complexity: O(SP)  where S, P is the length of s and p;
- Space Complexity: O(SP)

Top-Down Variation
```java
class Solution {
    private int[][] dp;
    public boolean isMatch(String s, String p) {
        int[][] memo = new int[s.length() + 1][p.length() + 1];
        for(int i = 0; i < s.length() + 1; i++)
            for(int j = 0; j < p.length() + 1; j++)
                memo[i][j] = -1;
        return dpMatch(0, 0, s, p, memo);
    }
    private boolean dpMatch(int i, int j, String s, String p, int[][] memo){
        if(memo[i][j] != -1) return memo[i][j] == 1;
        if (j == p.length()) return i == s.length();
        boolean res;
        boolean first_match = i < s.length() && (s.charAt(i) == p.charAt(j) || p.charAt(j) == '.');
        if (j + 1 < p.length() && p.charAt(j + 1) == '*')
            res = (first_match && dpMatch(i + 1, j, s, p, memo))
                || dpMatch(i, j + 2, s, p, memo);
        else
            res = first_match && dpMatch(i + 1, j + 1, s, p, memo);
        memo[i][j] = res ? 1 : 0;
        return res;
    }
    
   
}
```

Bottom-Up Variation
```java
class Solution {
    private int[][] dp;
    public boolean isMatch(String s, String p) {
        int slen = s.length(), plen = p.length();
        //相当于后面加一个空字符
        boolean[][] dp = new boolean[slen + 1][plen + 1];
        //dp[i][plen] = true(i == slen), false(i < slen)
        
        dp[slen][plen] = true;
        //因为p的符号有前后关系所以从尾部开始dp[i][j]代表s[i:]与p[j:]的匹配情况
        for(int i = slen; i >= 0; i--){
            for(int j = plen - 1; j >= 0; j--){
                //i == slen则p[j](j <= plen - 1)一定不可能与s[i](空字符)匹配
                boolean first_match = 
                    i < slen && (s.charAt(i) == p.charAt(j)
                    || p.charAt(j) =='.');
                //j + 1 < plen防止越界
                if (j + 1 < plen && p.charAt(j + 1) == '*')
                //dp[i][j + 2]不需要first_match因为这是与空字符匹配,
                //相当于first_match一定为true。
                //这两种情况对应p[j:j+1]匹配0个或至少一个s[i]
                    dp[i][j] = dp[i][j + 2] || (first_match && dp[i + 1][j]);
                else 
                    dp[i][j] = first_match && dp[i + 1][j + 1];
                
            }
        }
        return dp[0][0];
        
    }
    
   
}
```