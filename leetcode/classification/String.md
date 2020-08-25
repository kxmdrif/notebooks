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

## 20. Valid Parentheses

**soultion**

Approach: Stack

- Time Complexity: O(n)
- Space Complexity: O(1)

```java
class Solution {
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for(char c : s.toCharArray()){
            if (c == '(')
                stack.push(')');
            else if (c == '[')
                stack.push(']');
            else if (c == '{')
                stack.push('}');
            else if(stack.isEmpty() || stack.pop() != c)
                return false;
        }
        return stack.isEmpty();
    }
}
```

## 22. Generate Parentheses
**solution**
Approach: Backtracking

- Time Complexity: O(4^n / sqrt(n))
- Space Complexity: O(n)

```java
class Solution {
    
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        dfs(n, 0, 0, "", res);
        return res;
    }
    
    public void dfs(int n, int left, int right, String comb, List<String> res){
        //left is the number of '(', right is the number of ')'
        //if left = n and right = n, add to result list
        if (left == n && right == n){
            res.add(comb);
            return;
        }
        //the following condition can ensure that the combination is well-formed
        
        //if left < n , we can add '('
        if (left < n)
            dfs(n, left + 1, right, comb + '(', res);
        //if right < left, we can add ')'
        if (right < left)
            dfs(n, left, right + 1, comb + ')', res);
    }
}
```

## 32. Longest Valid Parentheses
**solution**

Approach 1: Using Stack

- Time Complexity: O(n)
- Space Complexity: O(n)

```java
class Solution {
    public int longestValidParentheses(String s) {
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        int maxlen = 0;
        for(int i = 0; i <s.length(); i++){
            if (s.charAt(i) == '(')
                stack.push(i);
            else{
                stack.pop();
                if (stack.isEmpty())
                    stack.push(i);
                maxlen = Math.max(maxlen, i - stack.peek());
                
            }
        }
        return maxlen;
    }
}
```
Approach 2: Without extra space

- Time Complexity: O(n)
- Space Complexity: O(1)

```java
public class Solution {
    public int longestValidParentheses(String s) {
        int left = 0, right = 0, maxlength = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                left++;
            } else {
                right++;
            }
            if (left == right) {
                maxlength = Math.max(maxlength, 2 * right);
            } else if (right >= left) {
                left = right = 0;
            }
        }
        left = right = 0;
        for (int i = s.length() - 1; i >= 0; i--) {
            if (s.charAt(i) == '(') {
                left++;
            } else {
                right++;
            }
            if (left == right) {
                maxlength = Math.max(maxlength, 2 * left);
            } else if (left >= right) {
                left = right = 0;
            }
        }
        return maxlength;
    }
}
```

## 49. Group Anagrams
**solution**

- Time complexity: O(NK), where N is the length of strs, and K is the maximum length of a string in strs
- Space complexity: O(NK)

```java

//Categorize by Count
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for(String str : strs){
            char[] arr = new char[26];
            for(char c : str.toCharArray())
                arr[c - 'a']++;
            
            String key = String.valueOf(arr);
            
            List<String> value = map.getOrDefault(key,
                                        new ArrayList<>());
            value.add(str);
            map.put(key, value);
            
        }
        return new ArrayList<>(map.values());
        
    }
}
```

## 76.  Minimum Window Substring

```java
class Solution {
    public String minWindow(String s, String t) {
        
        if (s.isEmpty() || t.isEmpty()) return "";
        
        char[] chs = s.toCharArray();
        char[] cht = t.toCharArray();
        
        int[] dictT = new int[128];
        int[] window = new int[128];
        for(char c : cht)
            dictT[c]++;
        
        //start, end初始值为0以及最后结果[start, end)
        //应对start, end在循环中从未更新的情况如s = "a", t = "aa"
        //min也不可少否则只有start和end两者差为0无法更新最小值
        int l = 0, r = 0, start = 0, end = 0;
        int min = Integer.MAX_VALUE;
        int required = cht.length;
        //formed 代表窗口里面所有在T中出现的字符数量
        //(每个字符数量小于等于T中数量,若超过则按T中数量算)的总和
        int formed = 0;
        
        while(r < chs.length){
            if (window[chs[r]]++ < dictT[chs[r]])
                formed++;
            while(l <= r && formed == required){
                if (r - l + 1 < min){
                    start = l;
                    end = r + 1;
                    min = r - l + 1;
                }
                if (window[chs[l]]-- <= dictT[chs[l]])
                    formed--;
                l++;
            }
            r++;
        }
        
        return s.substring(start, end);
        
    }
}
```

## 301. Remove Invalid Parentheses
**solution**

```java
class Solution {
    public List<String> removeInvalidParentheses(String s) {
        List<String> res = new ArrayList<>();
        remove(s, res, 0, 0, ')');
        return res;
    }
    
    private void remove(String s, List<String> res,
                        int iStart, int jStart, char ch){
        //each level remove one parentheses
        for(int count = 0, i = iStart; i < s.length(); i++){
            if (s.charAt(i) == '(' || s.charAt(i) == ')')
                count += (s.charAt(i) == ch ? -1 : 1);
            if (count >= 0) continue;
            // We have an extra closed paren we need to remove
            for(int j = jStart; j <= i; j++){
                // Try removing one at each position, skipping duplicates
                //example: (())) only remove index == 2
                if (s.charAt(j) == ch &&
                    (j == jStart || s.charAt(j - 1) != ch))
                    //After remove the char at pos j, the sbustring before iStart = i
                    //is valid, so we start at iStart on the next level of recursion.
                    //-------------------------------------------------------------
                    //jStart = j prevents duplicates(for example: 
                    //remove idx=a then remove idx=b and remove idx=b then remove
                    //idx = a will produce duplicate for a < b if we don't 
                    //start at the last removal positation
                    remove(s.substring(0, j) + s.substring(j + 1), res, i, j, ch);
            }
            return;// Stop here. The recursive calls handle the rest of the string.
        }
        
        // No invalid closed parenthesis detected.
        //Now check opposite direction, or reverse back to original direction.
        String reversed = new StringBuilder(s).reverse().toString();
        if (ch == ')')
            remove(reversed, res, 0, 0, '(');
        else//reverse two times => recover the original relative sequence
            res.add(reversed);
    }
}
```

## 394. Decode String
**solution**

```java
class Solution {
    public String decodeString(String s) {
        StringBuilder res = new StringBuilder();
        Stack<Integer> nums = new Stack<>();
        Stack<StringBuilder> strs = new Stack<>();
        int count = 0;
        //res stores the decode result until the current char
        for(char c : s.toCharArray()){
            if (Character.isDigit(c)){
                count = 10 * count + c - '0';
            }
            
            else if (c == '['){
                nums.push(count);
                count = 0;
                strs.push(res);
                res = new StringBuilder();
            }else if (c == ']'){
                int cnt = nums.pop();
                StringBuilder tmp = res;
                res = strs.pop();
                for(int i = 0; i < cnt; i++)
                    res.append(tmp);
            }else{
                res.append(c);
            }
            
        }
        return res.toString();
    }
}
```

## 438. Find All Anagrams in a String
**solution**

Approach: Similar To 76.  Minimum Window Substring

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        char[] chs = s.toCharArray();
        char[] chp = p.toCharArray();
        
        int[] dictP = new int[128];
        int[] window = new int[128];
        for(char c : chp)
            dictP[c]++;
        
        int l = 0, r = 0;
        int formed = 0, required = chp.length;
        List<Integer> res = new ArrayList<>();
        
        while(r < chs.length){
            if (window[chs[r]]++ < dictP[chs[r]])
                formed++;
            while(l <= r && formed == required){
                //if this condition(r - l + 1 == chp.length) is true
                //next time, the loop will end.
                //because in this case, window and dictP is same;
                if (r - l + 1 == chp.length)
                    res.add(l);
                if (window[chs[l]]-- <= dictP[chs[l]])
                    formed--;
                l++;
            }
            r++;
        }
        return res;
    }
}
```

## 621. Task Scheduler
**solution**
- Time Complexity: O(n)
- Space Complexity: O(26)


```java
class Solution {
    public int leastInterval(char[] tasks, int n) {
        int[] count = new int[26];
        for(char c : tasks)
            count[c - 'A']++;
        Arrays.sort(count);
        int idx = 25;
        while(idx >= 0 && count[idx] == count[25])
              idx--;
        return Math.max(tasks.length,
                        25 - idx + (n + 1) * (count[25] - 1));
        
    }
}
```

First consider the most frequent characters, we can determine their relative positions first and use them as a frame to insert the remaining less frequent characters. Here is a proof by construction:

Let F be the set of most frequent chars with frequency k.
We can create k chunks, each chunk is identical and is a string consists of chars in F in a specific fixed order.
Let the heads of these chunks to be H_i; then H_2 should be at least n chars away from H_1, and so on so forth; then we insert the less frequent chars into the gaps between these chunks sequentially one by one ordered by frequency in a decreasing order and try to fill the k-1 gaps as full or evenly as possible each time you insert a character. In summary, append the less frequent characters to the end of each chunk of the first k-1 chunks sequentially and round and round, then join the chunks and keep their heads' relative distance from each other to be at least n.

Examples:

AAAABBBEEFFGG 3

here X represents a space gap:
```
Frame: "AXXXAXXXAXXXA"
insert 'B': "ABXXABXXABXXA" <--- 'B' has higher frequency than the other characters, insert it first.
insert 'E': "ABEXABEXABXXA"
insert 'F': "ABEFABEXABFXA" <--- each time try to fill the k-1 gaps as full or evenly as possible.
insert 'G': "ABEFABEGABFGA"
```

AACCCBEEE 2
```
3 identical chunks "CE", "CE CE CE" <-- this is a frame
insert 'A' among the gaps of chunks since it has higher frequency than 'B' ---> "CEACEACE"
insert 'B' ---> "CEABCEACE" <----- result is tasks.length;
```

AACCCDDEEE 3
```
3 identical chunks "CE", "CE CE CE" <--- this is a frame.
Begin to insert 'A'->"CEA CEA CE"
Begin to insert 'B'->"CEABCEABCE" <---- result is tasks.length;
```

ACCCEEE 2
```
3 identical chunks "CE", "CE CE CE" <-- this is a frame
Begin to insert 'A' --> "CEACE CE" <-- result is (c[25] - 1) * (n + 1) + 25 -i = 2 * 3 + 2 = 8
```

## 647. Palindromic Substrings
**solution**

- Time Complexity: O(n^2)
- Space Complexity: O(1)

```java
class Solution {
    public int countSubstrings(String s) {
        int res = 0;
        for(int i = 0; i < s.length(); i++){
            res += appendCenter(s, i, i);
            res += appendCenter(s, i, i + 1);
        }
        return res;
    }
    
    private int appendCenter(String s, int l, int r){
        int count = 0;
        while(l >= 0 && r < s.length()
              && s.charAt(l) == s.charAt(r)){
            count++;
            l--;
            r++;
        }
        return count;
    }
}
```